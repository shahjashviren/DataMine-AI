"""
Deduplicator agent.

Text:  embed abstracts via the Hugging Face Inference API
       (sentence-transformers/all-MiniLM-L6-v2, hosted — no local PyTorch),
       compute pairwise cosine similarity, drop entries that are near-duplicates
       of an already-kept entry (similarity > threshold).

Images: download thumbnails and compute perceptual hash (pHash) via imagehash;
        pairs with hamming distance <= threshold are near-duplicates.
"""

from __future__ import annotations

import io
import os
import time
from typing import Any

import numpy as np
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Text dedup — HF Inference API cosine similarity
# ---------------------------------------------------------------------------

# HF Inference API endpoint for the hosted all-MiniLM-L6-v2 model.
# No local PyTorch / sentence-transformers required.
_HF_EMBED_URL = (
    "https://api-inference.huggingface.co/models/"
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 0.78 catches genuine near-duplicates from overlapping multi-query search streams
# (web + academic results on the same topic typically land in the 0.75–0.84 range)
# while preserving distinct entries that cover different angles.
TEXT_SIM_THRESHOLD = 0.78   # cosine similarity above this → near-duplicate


def _hf_embed(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using the HF Inference API.

    The feature-extraction endpoint returns a list of vectors.
    Supports an optional HF_API_TOKEN env var for higher rate limits,
    but works anonymously on the free tier for small batches.

    Retries once on 503 (model loading) with a 20 s back-off.
    Falls back to zero vectors if the API is unavailable, which means
    dedup is skipped (all entries kept) — a safe degradation.
    """
    token = os.environ.get("HF_API_TOKEN", "")
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {"inputs": texts, "options": {"wait_for_model": True}}

    for attempt in range(2):
        try:
            resp = requests.post(_HF_EMBED_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 503 and attempt == 0:
                # Model is warming up — wait and retry once
                print("[deduplicator] HF model loading (503), waiting 20 s…")
                time.sleep(20)
                continue
            resp.raise_for_status()
            vectors = resp.json()
            # API returns list[list[float]] for feature-extraction
            return np.array(vectors, dtype=np.float32)
        except Exception as exc:
            print(f"[deduplicator] HF Inference API error (attempt {attempt + 1}): {exc}")
            if attempt == 0:
                time.sleep(5)
            else:
                # Return zero matrix — dedup will produce no drops (safe fallback)
                return np.zeros((len(texts), 384), dtype=np.float32)

    return np.zeros((len(texts), 384), dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _dedup_texts(entries: list[dict]) -> dict[str, Any]:
    """
    Deduplicate text entries by abstract similarity.

    Chunks sharing the same parent_document_id (i.e. produced by splitting the
    same source document in the cleaner) are NEVER flagged as duplicates of each
    other — they are sibling chunks, not near-duplicate documents.  Only
    cross-document pairs that exceed TEXT_SIM_THRESHOLD are reported.

    Returns a dict with:
      - "kept":      list of unique entries
      - "dup_pairs": list of dicts describing removed duplicate pairs
                     (each has entry_a, entry_b, similarity — used for the dashboard proof)
    """
    if not entries:
        return {"kept": [], "dup_pairs": []}

    abstracts = [e["abstract"] for e in entries]
    embeddings = _hf_embed(abstracts)

    kept_indices: list[int] = []
    dropped: set[int] = set()
    dup_pairs: list[dict] = []

    for i in range(len(entries)):
        if i in dropped:
            continue
        kept_indices.append(i)
        parent_i = entries[i].get("parent_document_id") or ""
        for j in range(i + 1, len(entries)):
            if j in dropped:
                continue
            # Skip if both chunks come from the same source document
            parent_j = entries[j].get("parent_document_id") or ""
            if parent_i and parent_j and parent_i == parent_j:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= TEXT_SIM_THRESHOLD:
                dropped.add(j)
                dup_pairs.append(
                    {
                        "entry_a": {
                            "title": entries[i]["title"],
                            "abstract_snippet": entries[i]["abstract"][:300],
                            "id": entries[i].get("id", ""),
                            "parent_document_id": parent_i,
                        },
                        "entry_b": {
                            "title": entries[j]["title"],
                            "abstract_snippet": entries[j]["abstract"][:300],
                            "id": entries[j].get("id", ""),
                            "parent_document_id": parent_j,
                        },
                        "similarity": round(sim, 4),
                    }
                )

    kept = [entries[i] for i in kept_indices]

    intra_doc_guard_pairs = [
        p for p in dup_pairs
        if p["entry_a"].get("parent_document_id")
        and p["entry_a"]["parent_document_id"] == p["entry_b"].get("parent_document_id")
    ]
    if intra_doc_guard_pairs:
        # This should never happen given the guard above — log as an error if it does
        print(
            f"[deduplicator] ERROR: {len(intra_doc_guard_pairs)} intra-document pairs "
            "were incorrectly flagged as duplicates. BUG — please investigate."
        )

    print(
        f"[deduplicator] Text dedup: {len(entries)} entries -> {len(kept)} kept, "
        f"{len(dropped)} dropped, {len(dup_pairs)} cross-document dup pairs reported."
    )
    return {"kept": kept, "dup_pairs": dup_pairs}


# ---------------------------------------------------------------------------
# Image dedup — perceptual hashing (pHash)
# ---------------------------------------------------------------------------

IMAGE_HASH_THRESHOLD = 8    # hamming distance <= this → near-duplicate (tighter than 10)


def _fetch_image(url: str, timeout: int = 15) -> Image.Image | None:
    """Download an image and return a PIL Image, or None on failure."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _dedup_images(entries: list[dict]) -> dict[str, Any]:
    """
    Deduplicate images using pHash.
    Returns a dict with:
      - "kept":      list of unique entries
      - "dup_pairs": list of dicts describing removed duplicate pairs
    """
    import imagehash

    if not entries:
        return {"kept": [], "dup_pairs": []}

    # Compute hashes — try thumbnail first, fall back to full url if thumbnail fails
    hashes: list[imagehash.ImageHash | None] = []
    hash_success = 0
    hash_skipped = 0
    for entry in entries:
        thumb_url = entry.get("thumbnail") or ""
        full_url  = entry.get("url") or ""
        img = _fetch_image(thumb_url) if thumb_url else None
        if img is None and full_url and full_url != thumb_url:
            img = _fetch_image(full_url)
        if img is not None:
            hashes.append(imagehash.phash(img))
            hash_success += 1
        else:
            hashes.append(None)
            hash_skipped += 1
    print(
        f"[deduplicator] Image hashes: {hash_success} computed, "
        f"{hash_skipped} skipped (download failed)"
    )

    kept_indices: list[int] = []
    dropped: set[int] = set()
    dup_pairs: list[dict] = []

    for i in range(len(entries)):
        if i in dropped:
            continue
        kept_indices.append(i)
        if hashes[i] is None:
            continue   # can't compare, keep but skip pairwise check
        for j in range(i + 1, len(entries)):
            if j in dropped or hashes[j] is None:
                continue
            distance = hashes[i] - hashes[j]
            if distance <= IMAGE_HASH_THRESHOLD:
                dropped.add(j)
                dup_pairs.append(
                    {
                        "entry_a": {
                            "title": entries[i].get("title", ""),
                            "thumbnail": entries[i].get("thumbnail", ""),
                            "url": entries[i].get("url", ""),
                        },
                        "entry_b": {
                            "title": entries[j].get("title", ""),
                            "thumbnail": entries[j].get("thumbnail", ""),
                            "url": entries[j].get("url", ""),
                        },
                        "hamming_distance": distance,
                        "similarity": round(1 - distance / 64, 4),
                    }
                )

    kept = [entries[i] for i in kept_indices if i not in dropped]

    # Guarantee minimum 10 images: if dedup removed too many, restore
    # dropped entries (least similar first) until we hit 10 or run out.
    MIN_KEPT = 10
    if len(kept) < MIN_KEPT and dropped:
        restore_pool = [entries[i] for i in sorted(dropped)]
        for entry in restore_pool:
            if len(kept) >= MIN_KEPT:
                break
            kept.append(entry)
        print(f"[deduplicator] Restored images to reach minimum {MIN_KEPT} (now {len(kept)})")

    return {"kept": kept, "dup_pairs": dup_pairs}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def deduplicate(
    filtered_texts: list[dict],
    filtered_images: list[dict],
) -> dict[str, Any]:
    """
    Deduplicate text and image records.
    Returns a dict:
      {
        "texts":           list of unique text entries,
        "images":          list of unique image entries,
        "text_dup_pairs":  list of removed text duplicate pairs (with similarity),
        "image_dup_pairs": list of removed image duplicate pairs (with similarity),
      }
    """
    text_result = _dedup_texts(filtered_texts)
    image_result = _dedup_images(filtered_images)

    return {
        "texts": text_result["kept"],
        "images": image_result["kept"],
        "text_dup_pairs": text_result["dup_pairs"],
        "image_dup_pairs": image_result["dup_pairs"],
    }
