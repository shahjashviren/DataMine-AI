"""
Pipeline integration test — runs the full agent chain on a given topic
and writes pipeline.log + run_report.json to outputs/{run_id}/.

Usage:
    python pipeline_test.py "topic prompt here"
    python pipeline_test.py   # uses default topic
"""
import sys
import io
import json
import os
import uuid
import traceback

# Force UTF-8 output so the Windows console doesn't crash on special chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

from dotenv import load_dotenv
load_dotenv()

# Intercept all print() output into the log buffer
import builtins
log_lines: list[str] = []
_orig_print = builtins.print


def _log_print(*args, **kwargs):
    line = " ".join(str(a) for a in args)
    log_lines.append(line)
    _orig_print(*args, **kwargs)


builtins.print = _log_print

from agents.intent_parser import parse_intent
from agents.collector import collect_data
from agents.cleaner import clean_data
from agents.quality_filter import filter_quality
from agents.deduplicator import deduplicate
from agents.structurer import structure_output

PROMPT = sys.argv[1] if len(sys.argv) > 1 else (
    "I need training data about deep sea ocean exploration and marine geology"
)
RUN_ID = str(uuid.uuid4())

print("=" * 60)
print("PIPELINE TEST RUN")
print("run_id:", RUN_ID)
print("prompt:", PROMPT)
print("=" * 60)

try:
    # Step 1 — Intent
    topic, queries = parse_intent(PROMPT)
    print(f"[STEP1] topic={topic!r}  queries={queries}")

    # Step 2 — Collect
    raw_texts, raw_images = collect_data(topic, queries)
    print(f"[STEP2] raw_texts={len(raw_texts)}  raw_images={len(raw_images)}")

    wiki_count = sum(1 for t in raw_texts if t.get("source") == "wikipedia")
    s2_count   = sum(1 for t in raw_texts if t.get("source") == "semantic_scholar")
    print(f"  sources: Wikipedia={wiki_count}  SemanticScholar={s2_count}")

    # Step 3 — Clean
    clean_texts, clean_images = clean_data(raw_texts, raw_images)
    print(f"[STEP3] clean_texts={len(clean_texts)}  clean_images={len(clean_images)}")

    # Verify: clean_texts count is traceable (chunks tracked by parent_document_id)
    missing_parent = [e for e in clean_texts if not e.get("parent_document_id")]
    print(f"  missing parent_document_id: {len(missing_parent)} (must be 0)")
    assert not missing_parent, "BUG: entries missing parent_document_id"

    # Step 4 — Quality filter
    filtered_texts, filtered_images = filter_quality(clean_texts, clean_images, topic, PROMPT)
    print(f"[STEP4] filtered_texts={len(filtered_texts)}  filtered_images={len(filtered_images)}")

    # Step 5 — Dedup
    dedup_result = deduplicate(filtered_texts, filtered_images)
    print(
        f"[STEP5] dedup_texts={len(dedup_result['texts'])}  "
        f"dedup_images={len(dedup_result['images'])}  "
        f"text_dup_pairs={len(dedup_result['text_dup_pairs'])}"
    )

    # Verify no intra-doc dup pairs
    intra = [
        p for p in dedup_result["text_dup_pairs"]
        if p["entry_a"].get("parent_document_id")
        and p["entry_a"]["parent_document_id"] == p["entry_b"].get("parent_document_id")
    ]
    print(f"  intra-doc dup pairs: {len(intra)} (must be 0)")
    assert not intra, "BUG: intra-doc pairs reported"

    # Step 6 — Structure
    report = structure_output(RUN_ID, topic, clean_texts, clean_images, dedup_result)

    # Verify savings derivation is present
    savings = report.get("savings", {})
    print(
        f"[STEP6] baseline_gpu_hours={savings.get('baseline_gpu_hours')}  "
        f"derivation={savings.get('baseline_derivation')!r}"
    )
    assert "baseline_derivation" in savings, "BUG: baseline_derivation missing from savings"

except Exception as exc:
    print(f"[ERROR] Pipeline failed: {exc}")
    traceback.print_exc()

# Always write the log even on failure
out_dir = os.path.join("outputs", RUN_ID)
os.makedirs(out_dir, exist_ok=True)
log_path = os.path.join(out_dir, "pipeline.log")
with open(log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
print(f"pipeline.log -> {log_path}")

# Print run_report.json if it was written
report_path = os.path.join(out_dir, "run_report.json")
if os.path.exists(report_path):
    with open(report_path, encoding="utf-8") as f:
        report_data = json.load(f)
    report_display = {k: v for k, v in report_data.items() if k != "archive_path"}
    print()
    print("=" * 60)
    print("run_report.json")
    print("=" * 60)
    print(json.dumps(report_display, indent=2, ensure_ascii=False))
else:
    print("[WARN] run_report.json was not written (pipeline may have failed)")
