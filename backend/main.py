"""
DataMine AI — FastAPI backend
Orchestrates the 5-step agent pipeline and streams status events to the frontend.

Changes vs v1:
  - user_prompt is forwarded into filter_quality so the Groq rubric adapts
    dynamically to whatever the user described.
  - SSE keepalive pings prevent proxies / browsers from closing idle connections
    during long-running steps (e.g. arXiv multi-page fetch, bulk Groq filtering).
  - Large SSE payload (dedup_done dup-pairs) is trimmed before sending to avoid
    hitting proxy response-size limits; full data lives in the run report.
  - StreamingResponse explicitly sets Transfer-Encoding:chunked-friendly headers.
"""

import asyncio
import json
import math
import os
import uuid
from typing import AsyncGenerator

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agents.intent_parser import parse_intent
from agents.collector import collect_data
from agents.cleaner import clean_data
from agents.quality_filter import filter_quality
from agents.deduplicator import deduplicate
from agents.structurer import structure_output

app = FastAPI(
    title="DataMine AI",
    version="2.0.0",
    # Raise default body / response limits for large dataset payloads
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run store: run_id -> result dict
_runs: dict[str, dict] = {}

# How often to emit an SSE keepalive comment during blocking steps (seconds).
# Prevents nginx / Render / Vercel edge proxies from closing the connection.
_KEEPALIVE_INTERVAL = 20.0


class MineRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------

def _sanitize(obj: object) -> object:
    """
    Recursively convert NumPy scalar types (int64, float32, ndarray) and
    non-finite floats to JSON-safe native Python types.
    Called before every json.dumps() so SSE events never crash on int64.
    """
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


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _event(data: dict) -> str:
    """Format a dict as an SSE data line, sanitizing NumPy types first."""
    return f"data: {json.dumps(_sanitize(data))}\n\n"


def _keepalive() -> str:
    """SSE comment — keeps the TCP connection alive without triggering a client event."""
    return ": keepalive\n\n"


# ---------------------------------------------------------------------------
# Pipeline runner (async generator)
# ---------------------------------------------------------------------------

async def _run_pipeline(run_id: str, prompt: str) -> AsyncGenerator[str, None]:
    """Drive the full agent chain and yield SSE events at each stage."""

    def emit(step: str, message: str, payload: dict | None = None) -> str:
        event: dict = {"run_id": run_id, "step": step, "message": message}
        if payload:
            event.update(payload)
        return _event(event)

    # ── Reset / init all run-local state ─────────────────────────────────
    # Every variable used below is declared here so there is no possibility
    # of a previous run's value leaking in through a module-level name.
    topic: str = ""
    queries: list[str] = []
    raw_texts: list[dict] = []
    raw_images: list[dict] = []

    import logging
    logging.info("[pipeline] run_id=%s  prompt=%r  — all local state reset", run_id, prompt[:120])
    print(f"[pipeline] START run_id={run_id}  prompt={prompt[:120]!r}")

    # ── Step 1: Intent parsing ────────────────────────────────────────────
    yield emit("intent", "Parsing your prompt…")
    await asyncio.sleep(0)

    topic, queries = await asyncio.to_thread(parse_intent, prompt)
    print(f"[pipeline] run_id={run_id}  parsed topic={topic!r}  queries={queries}")
    yield emit("intent_done", f"Understood topic: {topic}",
               {"topic": topic, "queries": queries})

    # ── Step 2: Collect ───────────────────────────────────────────────────
    yield emit("collect", f"Collecting data for '{topic}' across 3 search vectors…")
    loop = asyncio.get_running_loop()
    _collect_fut = loop.run_in_executor(None, collect_data, topic, queries)
    while True:
        done, _ = await asyncio.wait({_collect_fut}, timeout=_KEEPALIVE_INTERVAL)
        if done:
            raw_texts, raw_images = _collect_fut.result()
            break
        yield _keepalive()
    yield emit(
        "collect_done",
        f"Collected {len(raw_texts)} text entries and {len(raw_images)} images.",
        {"raw_text_count": len(raw_texts), "raw_image_count": len(raw_images)},
    )

    # ── Step 3: Clean ─────────────────────────────────────────────────────
    yield emit("clean", "Cleaning raw data…")
    clean_texts, clean_images = await asyncio.to_thread(clean_data, raw_texts, raw_images)
    yield emit(
        "clean_done",
        f"{len(clean_texts)} text and {len(clean_images)} images after cleaning.",
        {"clean_text_count": len(clean_texts), "clean_image_count": len(clean_images)},
    )

    # ── Step 4: Quality filter (Groq, bulk + backoff) ─────────────────────
    yield emit("filter", f"Running quality filter on {len(clean_texts)} entries (Groq)…")
    _filter_fut = loop.run_in_executor(
        None, filter_quality, clean_texts, clean_images, topic, prompt
    )
    filtered_texts: list[dict] = []
    filtered_images: list[dict] = []
    while True:
        done, _ = await asyncio.wait({_filter_fut}, timeout=_KEEPALIVE_INTERVAL)
        if done:
            filtered_texts, filtered_images = _filter_fut.result()
            break
        yield _keepalive()
    yield emit(
        "filter_done",
        f"{len(filtered_texts)} text and {len(filtered_images)} images passed quality filter.",
        {"filtered_text_count": len(filtered_texts), "filtered_image_count": len(filtered_images)},
    )

    # ── Step 5: Deduplicate ───────────────────────────────────────────────
    yield emit("dedup", "Deduplicating…")
    dedup_result = await asyncio.to_thread(deduplicate, filtered_texts, filtered_images)

    # Trim dup-pairs to avoid oversized SSE frames (full pairs live in report)
    sse_text_pairs  = dedup_result.get("text_dup_pairs", [])[:3]
    sse_image_pairs = dedup_result.get("image_dup_pairs", [])[:3]
    yield emit(
        "dedup_done",
        f"{len(dedup_result['texts'])} text and {len(dedup_result['images'])} images after dedup.",
        {
            "final_text_count":  len(dedup_result["texts"]),
            "final_image_count": len(dedup_result["images"]),
            "text_dup_pairs":    sse_text_pairs,
            "image_dup_pairs":   sse_image_pairs,
        },
    )

    # ── Step 6: Structure output + Alpaca synthesis ───────────────────────
    yield emit(
        "structure",
        f"Synthesizing {len(dedup_result['texts'])} Alpaca instruction records via Groq, "
        "then writing dataset files…"
    )
    # Pass clean_texts/clean_images (post-chunking counts) so that raw_text in
    # the report reflects the chunk count entering dedup — always >= final_text.
    _struct_fut = loop.run_in_executor(
        None, structure_output, run_id, topic, clean_texts, clean_images, dedup_result
    )
    report = None  # type: ignore[assignment]
    while True:
        done, _ = await asyncio.wait({_struct_fut}, timeout=_KEEPALIVE_INTERVAL)
        if done:
            report = _struct_fut.result()
            break
        yield _keepalive()
    _runs[run_id] = report

    yield emit("done", "Pipeline complete!", {"report": report})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/mine")
async def mine(req: MineRequest):
    """Start a pipeline run and return a run_id for SSE streaming."""
    run_id = str(uuid.uuid4())
    return {"run_id": run_id}


@app.get("/mine/{run_id}/stream")
async def mine_stream(run_id: str, prompt: str):
    """
    Stream SSE events for a pipeline run.

    Uses chunked transfer encoding and explicit no-buffering headers so that
    Render, Railway, and nginx reverse-proxies forward each event immediately
    rather than buffering until the response ends.
    """
    return StreamingResponse(
        _run_pipeline(run_id, prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # nginx: disable proxy buffering
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
    )


@app.get("/mine/{run_id}/download")
async def download(run_id: str):
    """Download the finished dataset archive for a completed run."""
    report = _runs.get(run_id)
    if not report:
        raise HTTPException(status_code=404, detail="Run not found or not yet complete")
    archive_path = report.get("archive_path")
    if not archive_path or not os.path.exists(archive_path):
        raise HTTPException(status_code=404, detail="Dataset file not found")
    return FileResponse(
        archive_path,
        filename=f"dataset_{run_id}.zip",
        media_type="application/zip",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
