"""
Structurer agent — Alpaca instruction-synthesis edition.

Upgrades over v1:
  - _synthesize_instructions(): passes deduped text chunks through Groq
    (llama-3.3-70b-versatile) to generate standard Alpaca fine-tuning records:
    {"instruction": "...", "input": "...", "output": "..."}
  - texts.jsonl now contains Alpaca records, not raw abstracts.
  - run_report sample_texts shows instruction/output fields.
  - Savings computation is unchanged — still derived from raw vs final counts.
"""

from __future__ import annotations

import json
import math
import os
import time
import random
import zipfile
from datetime import datetime, timezone
from typing import Any

from groq import Groq, RateLimitError

# ---------------------------------------------------------------------------
# Groq client (lazy init)
# ---------------------------------------------------------------------------

_groq_client: Groq | None = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Savings computation (unchanged)
# ---------------------------------------------------------------------------

A100_TDP_WATTS = 400.0
A100_TDP_KW = A100_TDP_WATTS / 1000.0
CLOUD_GPU_RATE_USD_PER_HOUR = 2.49
CARBON_INTENSITY_G_CO2_PER_KWH = 386.0
WATER_L_PER_KWH = 1.8

# Baseline GPU hours are derived from the actual raw entry count rather than
# a fixed constant.  Formula: assume training on the full uncleaned dataset
# takes 1 GPU-hour per 100 raw entries (A100 @ 400 W, batch fine-tuning).
# This makes the savings figures directly proportional to this run's real volume.
_GPU_HOURS_PER_100_ENTRIES = 1.0


def _derive_baseline_gpu_hours(raw_count: int) -> float:
    """
    Derive the baseline GPU hours from the actual raw entry count.
    Formula: raw_count / 100 * _GPU_HOURS_PER_100_ENTRIES
    Minimum of 0.1 h so the field is never exactly 0 for very small runs.
    """
    return max(0.1, round(raw_count / 100.0 * _GPU_HOURS_PER_100_ENTRIES, 4))


def compute_savings(raw_count: int, final_count: int) -> dict[str, Any]:
    """
    Compute resource savings from deduplication.

    baseline_gpu_hours is derived from raw_count (not a fixed constant):
      baseline_gpu_hours = raw_count / 100 * {_GPU_HOURS_PER_100_ENTRIES} GPU-h/100-entries
    gpu_hours_saved      = (duplicates_removed / raw_count) * baseline_gpu_hours
    energy_saved_kwh     = gpu_hours_saved * A100_TDP_KW  ({A100_TDP_KW} kW)
    cost_saved_usd       = gpu_hours_saved * ${CLOUD_GPU_RATE_USD_PER_HOUR}/h
    carbon_saved_kg_co2  = energy_saved_kwh * {CARBON_INTENSITY_G_CO2_PER_KWH} g/kWh / 1000
    water_saved_litres   = energy_saved_kwh * {WATER_L_PER_KWH} L/kWh
    """
    baseline_gpu_hours = _derive_baseline_gpu_hours(raw_count)

    if raw_count == 0:
        return {
            "data_reduction_pct": 0.0,
            "duplicates_removed": 0,
            "gpu_hours_saved": 0.0,
            "energy_saved_kwh": 0.0,
            "cost_saved_usd": 0.0,
            "carbon_saved_kg_co2": 0.0,
            "water_saved_litres": 0.0,
            "baseline_gpu_hours": baseline_gpu_hours,
            "baseline_derivation": f"max(0.1, {raw_count}/100 * {_GPU_HOURS_PER_100_ENTRIES}) = {baseline_gpu_hours}",
        }

    duplicates_removed = max(0, raw_count - final_count)
    data_reduction_pct = duplicates_removed / raw_count

    gpu_hours_saved  = data_reduction_pct * baseline_gpu_hours
    energy_saved_kwh = gpu_hours_saved * A100_TDP_KW
    cost_saved_usd   = gpu_hours_saved * CLOUD_GPU_RATE_USD_PER_HOUR
    carbon_saved_kg  = (energy_saved_kwh * CARBON_INTENSITY_G_CO2_PER_KWH) / 1000.0
    water_saved_l    = energy_saved_kwh * WATER_L_PER_KWH

    return {
        "data_reduction_pct": round(data_reduction_pct * 100, 2),
        "duplicates_removed": duplicates_removed,
        "gpu_hours_saved": round(gpu_hours_saved, 4),
        "energy_saved_kwh": round(energy_saved_kwh, 4),
        "cost_saved_usd": round(cost_saved_usd, 4),
        "carbon_saved_kg_co2": round(carbon_saved_kg, 4),
        "water_saved_litres": round(water_saved_l, 4),
        "baseline_gpu_hours": round(baseline_gpu_hours, 4),
        "baseline_derivation": (
            f"max(0.1, {raw_count}/100 * {_GPU_HOURS_PER_100_ENTRIES} GPU-h/100-entries) "
            f"= {baseline_gpu_hours} h"
        ),
    }


# ---------------------------------------------------------------------------
# Alpaca instruction synthesis
# ---------------------------------------------------------------------------

_SYNTH_MODEL  = "llama-3.3-70b-versatile"
_SYNTH_BATCH  = 8    # entries per Groq synthesis call (generation uses more tokens)
_MAX_RETRIES  = 4
_BACKOFF_BASE = 2.0
_BACKOFF_JITTER = 0.3
_RATE_LIMIT_PAUSE = 60.0

_SYNTH_SYSTEM = """\
You are an expert dataset curator creating instruction-tuning data for LLM fine-tuning.

Given a list of text excerpts (each with a title and content), produce a JSON array
where each element is an Alpaca-format training record:
{
  "instruction": "<a clear, answerable question or task derived from the excerpt>",
  "input": "<the relevant excerpt text, verbatim or lightly paraphrased, max 400 chars>",
  "output": "<a complete, factual answer or explanation based solely on the excerpt>"
}

Rules:
- The instruction must be a genuine question a learner would ask about this topic.
- The output must be grounded in the provided excerpt — do NOT hallucinate facts.
- Keep instruction ≤ 120 characters, output ≤ 300 characters.
- Return ONLY the JSON array — no markdown, no preamble, no explanation.
"""


def _synth_groq_call(entries_payload: str) -> str:
    """Call Groq with backoff; return raw response string."""
    client = _get_groq()
    delay  = _BACKOFF_BASE

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=_SYNTH_MODEL,
                messages=[
                    {"role": "system", "content": _SYNTH_SYSTEM},
                    {"role": "user",   "content": entries_payload},
                ],
                temperature=0.4,
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            wait = _RATE_LIMIT_PAUSE + random.uniform(0, _BACKOFF_JITTER)
            print(f"[structurer] Rate-limited. Waiting {wait:.1f}s (attempt {attempt})…")
            time.sleep(wait)
        except Exception as exc:
            wait = delay + random.uniform(0, _BACKOFF_JITTER)
            print(f"[structurer] Groq error: {exc}. Retry in {wait:.1f}s…")
            time.sleep(wait)
            delay *= 2.0

    return ""


def _fallback_alpaca(entry: dict) -> dict:
    """Template-based Alpaca record used when Groq synthesis fails."""
    title    = entry.get("title", "Topic")
    abstract = entry.get("abstract", "")
    return {
        "instruction": f"What is known about: {title[:100]}?",
        "input":       abstract[:400],
        "output":      abstract[400:800] if len(abstract) > 400 else abstract,
        "_source":     entry.get("source", ""),
        "_search_term": entry.get("search_term", ""),
    }


def _synthesize_instructions(entries: list[dict], topic: str) -> list[dict]:
    """
    Convert cleaned text chunks into Alpaca-format instruction-tuning records
    using Groq (llama-3.3-70b-versatile).

    Batches of _SYNTH_BATCH entries are sent per call.
    On parse failure, the batch falls back to template-based records.
    Returns a list of Alpaca dicts (same length as input entries).
    """
    if not entries:
        return []

    alpaca_records: list[dict] = []

    for batch_start in range(0, len(entries), _SYNTH_BATCH):
        batch = entries[batch_start : batch_start + _SYNTH_BATCH]

        # Build the payload: numbered list of title + excerpt
        lines = []
        for i, e in enumerate(batch, 1):
            snippet = e.get("abstract", "")[:500]
            lines.append(f'{i}. Title: {e.get("title","")}\nContent: {snippet}')
        payload = "\n\n".join(lines)

        raw = _synth_groq_call(payload)

        # Try to parse JSON array from response
        parsed: list[dict] | None = None
        try:
            clean = raw.strip()
            # Strip markdown fences if present
            clean = __import__("re").sub(r"```(?:json)?|```", "", clean).strip()
            parsed = json.loads(clean)
            if not isinstance(parsed, list):
                parsed = None
        except (json.JSONDecodeError, Exception):
            parsed = None

        if parsed and len(parsed) == len(batch):
            # Attach provenance metadata
            for record, entry in zip(parsed, batch):
                record["_source"]      = entry.get("source", "")
                record["_search_term"] = entry.get("search_term", "")
            alpaca_records.extend(parsed)
        else:
            # Fallback for failed batch
            print(
                f"[structurer] Synthesis parse failed for batch "
                f"{batch_start // _SYNTH_BATCH + 1} — using template fallback."
            )
            alpaca_records.extend(_fallback_alpaca(e) for e in batch)

        batch_num = batch_start // _SYNTH_BATCH + 1
        total_batches = (len(entries) + _SYNTH_BATCH - 1) // _SYNTH_BATCH
        print(f"[structurer] Synthesis batch {batch_num}/{total_batches} complete.")

    return alpaca_records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _output_dir(run_id: str) -> str:
    base = os.environ.get("OUTPUT_DIR", "outputs")
    path = os.path.join(base, run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _write_alpaca_jsonl(path: str, records: list[dict]) -> None:
    """Write Alpaca-format records as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(_sanitize(rec), ensure_ascii=False) + "\n")


def _write_image_metadata(path: str, images: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(images), f, ensure_ascii=False, indent=2)


def _create_zip(out_dir: str, run_id: str, files: list[str]) -> str:
    zip_path = os.path.join(out_dir, f"dataset_{run_id}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            if os.path.exists(fp):
                zf.write(fp, arcname=os.path.basename(fp))
    return zip_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def structure_output(
    run_id: str,
    topic: str,
    raw_texts: list[dict],
    raw_images: list[dict],
    dedup_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Synthesize Alpaca instruction records from deduped chunks, write output
    files, and return the full run report dict.
    """
    out_dir = _output_dir(run_id)

    final_texts  = dedup_result["texts"]
    final_images = dedup_result["images"]

    # --- Alpaca instruction synthesis ---
    print(f"[structurer] Synthesizing Alpaca records for {len(final_texts)} chunks…")
    alpaca_records = _synthesize_instructions(final_texts, topic)

    # --- Write Alpaca JSONL ---
    text_jsonl_path = os.path.join(out_dir, "texts.jsonl")
    _write_alpaca_jsonl(text_jsonl_path, alpaca_records)

    # --- Write image metadata ---
    images_meta_path = os.path.join(out_dir, "images_metadata.json")
    _write_image_metadata(images_meta_path, final_images)

    # --- Savings ---
    # raw_texts here are the post-cleaning chunks (always >= final_texts after dedup)
    raw_total   = len(raw_texts) + len(raw_images)
    final_total = len(final_texts) + len(final_images)
    savings = compute_savings(raw_total, final_total)

    # --- Run report ---
    report: dict[str, Any] = _sanitize({
        "run_id":    run_id,
        "topic":     topic,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "raw_text":        len(raw_texts),   # post-cleaning chunks entering dedup
            "raw_image":       len(raw_images),
            "final_text":      len(final_texts), # surviving chunks after dedup
            "final_image":     len(final_images),
            "alpaca_records":  len(alpaca_records),
        },
        "savings": savings,
        "text_dup_pairs":  dedup_result.get("text_dup_pairs",  [])[:5],
        "image_dup_pairs": dedup_result.get("image_dup_pairs", [])[:5],
        # Sample shows Alpaca format so the dashboard demonstrates real output
        "sample_texts": [
            {
                "instruction": r.get("instruction", ""),
                "input":       r.get("input", "")[:200],
                "output":      r.get("output", "")[:200],
            }
            for r in alpaca_records[:5]
        ],
        "sample_images": [
            {
                "title":     e.get("title", ""),
                "thumbnail": e.get("thumbnail") or e.get("url", ""),
                "license":   e.get("license", "unknown"),
                "source":    e.get("source", ""),
            }
            for e in final_images[:12]
        ],
    })

    # --- Write report JSON ---
    report_path = os.path.join(out_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- ZIP archive ---
    zip_path = _create_zip(out_dir, run_id, [text_jsonl_path, images_meta_path, report_path])
    report["archive_path"] = zip_path

    return report
