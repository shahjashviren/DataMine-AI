"""
Quality-filter agent — strict specificity edition.

Key design:
  - Text: YES/NO specificity gate per entry via Groq. The prompt asks whether
    the entry is SPECIFICALLY and DIRECTLY about the topic, not merely about a
    broader category or adjacent technology. Entries that get NO are dropped.
  - Images: heuristic pre-filter (domain, URL, license) THEN a Groq YES/NO
    relevance check on the image title/tags — tangential images (memorials,
    ceremonies, generic infrastructure) are removed.
  - Exponential backoff on Groq rate limits.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import random
from urllib.parse import urlparse

from groq import Groq, RateLimitError

_client: Groq | None = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_LLM_MODEL = "llama-3.1-8b-instant"   # fast, low-cost for bulk filtering

# Text: batch size for Groq calls
TEXT_BATCH_SIZE  = 20
# Image: batch size for Groq title-relevance calls
IMAGE_BATCH_SIZE = 30

MAX_RETRIES       = 5
BACKOFF_BASE      = 2.0
BACKOFF_JITTER    = 0.5
RATE_LIMIT_PAUSE  = 62.0   # fallback if no Retry-After header

# In-process cache: SHA-256(prompt+user_msg) -> raw YES/NO response string
# Skips Groq entirely for batches already judged in this process lifetime.
_filter_cache: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Image heuristic constants (pre-filter before Groq)
# ---------------------------------------------------------------------------

_BLOCKED_DOMAINS = {
    "shutterstock.com", "gettyimages.com", "istockphoto.com",
    "alamy.com", "dreamstime.com", "depositphotos.com",
    "123rf.com", "bigstockphoto.com", "stockfresh.com",
}

_BLOCKED_URL_PATTERNS = {
    "portrait", "headshot", "logo", "avatar", "profile-pic",
    "profile_pic", "thumbnail-person", "stock-photo", "stock_photo",
    "banner", "icon-", "icon_", "favicon",
}

_BLOCKED_CONTENT_WORDS = {
    "portrait", "headshot", "selfie", "stock photo", "profile photo",
    "couple", "wedding", "logo icon",
}

ACCEPTED_LICENSES = {
    "cc0", "pdm", "by", "by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd",
    "cc-by-sa", "cc-by", "unknown",
}


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        _client = Groq(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Groq call with exponential back-off
# ---------------------------------------------------------------------------

def _cache_key(messages: list[dict]) -> str:
    combined = "||".join(m.get("content", "") for m in messages)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _groq_with_backoff(messages: list[dict], max_tokens: int = 512) -> str:
    # Cache check — skip Groq entirely if this exact batch was already judged
    key = _cache_key(messages)
    if key in _filter_cache:
        return _filter_cache[key]

    client = _get_client()
    delay  = BACKOFF_BASE

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=_LLM_MODEL,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content or ""
            _filter_cache[key] = result
            return result

        except RateLimitError as exc:
            retry_after = RATE_LIMIT_PAUSE
            try:
                raw_headers = getattr(exc, "response", None)
                if raw_headers is not None:
                    ra = raw_headers.headers.get("Retry-After") or raw_headers.headers.get("x-ratelimit-reset-requests")
                    if ra:
                        retry_after = float(ra) + 1.0
            except Exception:
                pass
            wait = retry_after + random.uniform(0, BACKOFF_JITTER)
            print(
                f"[quality_filter] Rate-limited (429). Waiting {wait:.1f}s "
                f"(attempt {attempt}/{MAX_RETRIES})..."
            )
            time.sleep(wait)

        except Exception as exc:
            jitter = random.uniform(0, BACKOFF_JITTER)
            wait   = delay + jitter
            print(
                f"[quality_filter] Groq error: {exc}. "
                f"Retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})..."
            )
            time.sleep(wait)
            delay *= 2.0

    print("[quality_filter] All retries exhausted — defaulting entire batch to KEEP.")
    return ""


# ---------------------------------------------------------------------------
# Text filtering — strict YES/NO specificity gate
# ---------------------------------------------------------------------------

def _build_text_system_prompt(topic: str, user_prompt: str) -> str:
    return f"""\
You are a strict training-data relevance judge.

The user wants training data specifically about: "{topic}"
Their original request: \"\"\"{user_prompt}\"\"\"

For each numbered text excerpt below, answer YES or NO on its own line:
  YES — the text is SPECIFICALLY and DIRECTLY about "{topic}", not merely about a
        broader category, an adjacent technology, or general infrastructure that
        could apply to many unrelated fields.
  NO  — the text is off-topic, uses the same words in a different domain, or is
        only tangentially related (e.g. a broader survey where "{topic}" is one
        minor example among many unrelated examples).

Rules:
- If the text is about a broader parent category but not specifically "{topic}", answer NO.
- If the text uses the same keyword in a completely different field (maths, robotics, CS
  algorithm names, brand names), answer NO.
- Answer ONLY with the entry number and YES or NO, one per line, in order.
- No explanation, no punctuation beyond the number and YES/NO.

Example output for 4 entries:
1 YES
2 NO
3 YES
4 NO
"""


def _parse_yes_no(raw: str, batch_size: int) -> list[bool]:
    """
    Parse YES/NO lines like:
        1 YES
        2 NO
    Returns list of bools (True = keep). Defaults to True (keep) on parse failure.
    """
    keeps = [True] * batch_size   # fail-open: default keep on parse failure
    for line in raw.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                idx    = int(parts[0]) - 1   # 1-based to 0-based
                answer = parts[1].upper()
                if 0 <= idx < batch_size:
                    keeps[idx] = (answer == "YES")
            except (ValueError, IndexError):
                pass
    return keeps


def _topic_keywords(topic: str) -> list[str]:
    """Extract significant lowercase keywords from the topic string."""
    stop = {
        "need", "data", "about", "with", "from", "that", "this", "have",
        "what", "which", "when", "where", "will", "more", "into", "some",
        "them", "then", "than", "there", "their", "these", "training",
        "research", "study", "studies", "using", "based", "used", "both",
    }
    words = re.split(r"\W+", topic.lower())
    return [w for w in words if len(w) > 3 and w not in stop]


def _has_topic_keyword(entry: dict, keywords: list[str]) -> bool:
    if not keywords:
        return True
    haystack = f"{entry.get('title', '')} {entry.get('abstract', '')}".lower()
    return any(kw in haystack for kw in keywords)


def _filter_texts(entries: list[dict], topic: str, user_prompt: str) -> list[dict]:
    # Step 0: Hard keyword gate — reject entries with zero topic keyword matches.
    keywords = _topic_keywords(topic)
    keyword_passed  = [e for e in entries if _has_topic_keyword(e, keywords)]
    keyword_dropped = len(entries) - len(keyword_passed)
    if keyword_dropped:
        print(
            f"[quality_filter] Hard keyword gate: dropped {keyword_dropped} entries "
            "(no topic keyword in title+abstract)."
        )
    if not keyword_passed:
        return []

    # Step 1: Groq YES/NO specificity gate — all entries, no auto-pass on length.
    system_prompt = _build_text_system_prompt(topic, user_prompt)
    kept: list[dict] = []

    for batch_start in range(0, len(keyword_passed), TEXT_BATCH_SIZE):
        batch = keyword_passed[batch_start : batch_start + TEXT_BATCH_SIZE]

        user_lines = []
        for i, entry in enumerate(batch, 1):
            snippet = entry["abstract"][:400].replace("\n", " ")
            user_lines.append(
                f"{i}. Title: {entry['title']}\n   Content: {snippet}"
            )
        user_msg = "\n\n".join(user_lines)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        raw_response = _groq_with_backoff(messages, max_tokens=TEXT_BATCH_SIZE * 8)
        verdicts     = _parse_yes_no(raw_response, len(batch))

        for entry, keep in zip(batch, verdicts):
            status = "KEEP" if keep else "DROP"
            print(f"[quality_filter]   [{status}] {entry['title'][:70]}")
            if keep:
                kept.append(entry)

        kept_n = sum(verdicts)
        print(
            f"[quality_filter] Batch {batch_start // TEXT_BATCH_SIZE + 1}: "
            f"{kept_n}/{len(batch)} passed specificity gate"
        )

    return kept


# ---------------------------------------------------------------------------
# Image filtering — heuristic pre-filter + Groq title relevance check
# ---------------------------------------------------------------------------

def _image_passes_heuristics(entry: dict) -> bool:
    """Pre-filter: domain, URL pattern, blocked content words, license."""
    url   = (entry.get("url") or "").strip()
    title = (entry.get("title") or "").lower()

    if not url:
        return False

    try:
        domain = urlparse(url.lower()).netloc.lstrip("www.")
        if any(blocked in domain for blocked in _BLOCKED_DOMAINS):
            return False
    except Exception:
        pass

    url_path = urlparse(url.lower()).path
    if any(pat in url_path for pat in _BLOCKED_URL_PATTERNS):
        return False

    if title.strip() and any(bw in title for bw in _BLOCKED_CONTENT_WORDS):
        return False

    lic = entry.get("license", "unknown").lower().replace("cc-", "").strip()
    if lic not in ACCEPTED_LICENSES:
        return False

    return True


def _build_image_system_prompt(topic: str) -> str:
    return f"""\
You are a strict image-relevance judge for a training dataset about "{topic}".

For each numbered image title/description below, answer YES or NO:
  YES — the image is directly and specifically about "{topic}" (e.g. a photo of
        the actual subject, a diagram illustrating it, a scene where it is the
        primary focus).
  NO  — the image is tangential: a memorial event, an award ceremony, a generic
        landscape, generic infrastructure not specifically tied to "{topic}", a
        portrait of a person with only a vague connection, or anything where the
        link to "{topic}" is indirect or coincidental.

Answer ONLY the entry number and YES or NO, one per line, in order.
No explanation, no punctuation beyond the number and YES/NO.

Example output for 3 entries:
1 YES
2 NO
3 YES
"""


def _filter_images(entries: list[dict], topic: str) -> list[dict]:
    # Step 1: heuristic pre-filter (domain/URL/license) — free, fast
    heuristic_passed  = [e for e in entries if _image_passes_heuristics(e)]
    heuristic_dropped = len(entries) - len(heuristic_passed)
    if heuristic_dropped:
        print(
            f"[quality_filter] Image heuristic pre-filter: "
            f"dropped {heuristic_dropped}/{len(entries)}"
        )

    if not heuristic_passed:
        return []

    # Step 2: Groq YES/NO title relevance check
    # Only images with a non-empty title are sent; untitled images pass through
    # (Openverse/Wikimedia images without titles are assumed on-topic since they
    # were retrieved by querying the topic string directly).
    titled   = [e for e in heuristic_passed if (e.get("title") or "").strip()]
    untitled = [e for e in heuristic_passed if not (e.get("title") or "").strip()]

    kept_titled: list[dict] = []
    system_prompt = _build_image_system_prompt(topic)

    for batch_start in range(0, len(titled), IMAGE_BATCH_SIZE):
        batch = titled[batch_start : batch_start + IMAGE_BATCH_SIZE]
        lines = []
        for i, img in enumerate(batch, 1):
            title_text = (img.get("title") or "").strip()
            tags_text  = ", ".join(img.get("tags", [])[:5])
            desc       = title_text
            if tags_text:
                desc += f" [tags: {tags_text}]"
            lines.append(f"{i}. {desc}")
        user_msg = "\n".join(lines)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg},
        ]
        raw_response = _groq_with_backoff(messages, max_tokens=IMAGE_BATCH_SIZE * 8)
        verdicts     = _parse_yes_no(raw_response, len(batch))

        for img, keep in zip(batch, verdicts):
            status = "KEEP" if keep else "DROP"
            print(
                f"[quality_filter]   [img {status}] {(img.get('title') or '')[:60]}"
            )
            if keep:
                kept_titled.append(img)

        kept_n = sum(verdicts)
        print(
            f"[quality_filter] Image batch "
            f"{batch_start // IMAGE_BATCH_SIZE + 1}: "
            f"{kept_n}/{len(batch)} titled images passed"
        )

    kept = kept_titled + untitled
    print(
        f"[quality_filter] Images: {len(kept_titled)} titled kept + "
        f"{len(untitled)} untitled passed through = {len(kept)} total"
    )
    return kept


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def filter_quality(
    clean_texts: list[dict],
    clean_images: list[dict],
    topic: str,
    user_prompt: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    Quality-filter cleaned text and image records.

    Text:   hard keyword gate, then Groq YES/NO specificity check per entry.
            Entries that are only tangentially related or use the topic keyword
            in a different field are dropped.
    Images: heuristic pre-filter (domain/URL/license), then Groq YES/NO
            relevance check on image title + tags.  Untitled images pass through.
    """
    filtered_texts  = _filter_texts(clean_texts, topic, user_prompt or topic)
    filtered_images = _filter_images(clean_images, topic)
    return filtered_texts, filtered_images
