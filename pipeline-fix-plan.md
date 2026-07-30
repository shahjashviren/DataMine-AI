# Pipeline Fix Plan — Data Volume, Accuracy & Bug Fixes

## Top-Level Overview

**Goal:** Fix 5 critical bugs causing blank UI output and near-zero deduplication,
increase raw data volume meaningfully, and ensure the full pipeline produces
a populated, accurate run report that the frontend can display correctly.

**Scope:** `agents/quality_filter.py`, `agents/collector.py`,
`agents/deduplicator.py`, `agents/structurer.py`, `frontend/src/app/page.tsx`.
No new routes, no new frameworks, no infrastructure changes.

**Root cause summary (confirmed from run reports):**
1. Frontend `sample_texts` interface reads `title`/`abstract_snippet` but structurer now
   writes Alpaca records (`instruction`/`input`/`output`) — all text samples render blank.
2. Quality filter is over-aggressive: 30 raw entries → 2 survive (93% loss rate).
   The Groq `llama-3.1-8b-instant` model systematically scores short DDG snippets 1–4.
3. Deduplication finds 0 text duplicates in most runs — cosine threshold 0.86 is too strict
   for the actual overlap level produced by 3 queries.
4. Image dedup always finds 0 pairs — image downloads during dedup are silently failing
   (confirmed by timeout behavior with external URLs).
5. Collector returns only ~30 entries when it should return up to 105 — DDG limits and
   Semantic Scholar timeouts are silently capping results.

---

## Sub-Tasks

---

### Sub-Task 1 — Fix Frontend `sample_texts` Type Mismatch (Critical — UI is broken)

**Intent**
The `Report` interface in `page.tsx` expects `{ title, abstract_snippet }` for sample texts,
but `structurer.py` now emits Alpaca records `{ instruction, input, output }`.
This causes both sample text boxes to render blank in the Run Report dashboard.
Fix by updating the frontend interface and rendering to match the Alpaca format the backend
actually produces.

**Expected Outcomes**
- Sample text entries in the dashboard show `instruction` (as heading) and `output` (as body).
- The two blank text boxes in the screenshot become populated with real content.
- No change to the backend data format — frontend adapts to what structurer already emits.

**Todo List**
1. In `frontend/src/app/page.tsx`, update the `Report` interface:
   - Change `sample_texts: { title: string; abstract_snippet: string }[]`
     to `sample_texts: { instruction: string; input: string; output: string }[]`
2. In the "Sample text entries" render section (around line 486–498), update the JSX:
   - Replace `{t.title}` with `{t.instruction}`
   - Replace `{t.abstract_snippet}` with `{t.output}`
   - Optionally add a small "context" sub-line showing `{t.input}` truncated.
3. Add a `structure_done` step label to `STEP_LABELS` so the pipeline log shows a
   completion marker for the structuring step (currently missing).

**Relevant Context**
- [`frontend/src/app/page.tsx:51`](frontend/src/app/page.tsx:51) — `sample_texts` interface field
- [`frontend/src/app/page.tsx:482–499`](frontend/src/app/page.tsx:482) — sample text render block
- [`backend/agents/structurer.py:326–333`](backend/agents/structurer.py:326) — what actually gets written to `sample_texts`

**Status:** [ ] pending

---

### Sub-Task 2 — Fix Quality Filter Over-Rejection

**Intent**
The quality filter is eliminating 90%+ of entries — leaving 2 out of 30 in the screenshot.
The fix has two parts:
(a) Auto-pass short web snippets (DDG results, ≤ 300 chars) without sending them to Groq —
    they are valid context blocks and the LLM penalizes them for brevity regardless of content.
(b) Lower `SCORE_THRESHOLD` from `5` to `4` for the scored entries, matching the actual
    rubric wording in the prompt ("3–4 = tangentially related, borderline") — threshold 5
    cuts off the entire "usable" range.

**Expected Outcomes**
- At least 60–70% of collected entries survive the quality filter (not 6%).
- Short DDG web snippets (≤ 300 chars) are passed directly without a Groq API call.
- Groq is still called for longer entries where scoring is meaningful.
- `SCORE_THRESHOLD` comment in code is updated to match actual rubric boundaries.

**Todo List**
1. In `agents/quality_filter.py`, add a pre-pass in `_filter_texts()`:
   - Before batching for Groq, split entries into `short_entries` (abstract ≤ 300 chars)
     and `long_entries` (abstract > 300 chars).
   - `short_entries` are automatically added to `kept` — no Groq call.
   - Only `long_entries` go through the Groq batch scoring loop.
2. Lower `SCORE_THRESHOLD` from `5` to `4` in `quality_filter.py`.
3. Update the system prompt comment ("Entries scoring 5 or above" → "Entries scoring 4 or
   above") to stay consistent.
4. Add a `print` log line reporting how many entries were auto-passed as short snippets
   vs scored by Groq, so the behaviour is visible in server logs.

**Relevant Context**
- [`backend/agents/quality_filter.py:33`](backend/agents/quality_filter.py:33) — `SCORE_THRESHOLD = 5`
- [`backend/agents/quality_filter.py:193–234`](backend/agents/quality_filter.py:193) — `_filter_texts()` loop
- [`backend/agents/quality_filter.py:113`](backend/agents/quality_filter.py:113) — system prompt line "Entries scoring 5 or above"

**Status:** [ ] pending

---

### Sub-Task 3 — Fix Text Deduplication Threshold (0 Duplicates Found)

**Intent**
Nearly all runs show `text_dup_pairs: []` and `duplicates_removed: 0`. The `TEXT_SIM_THRESHOLD`
of `0.86` is too strict — overlapping queries produce pairs with cosine similarity in the
0.75–0.84 range which are real near-duplicates but never caught. Lower to `0.78` to catch
genuine overlap. This also makes the savings metric non-zero, which is essential for demo.

**Expected Outcomes**
- Runs on topics with 3 overlapping queries consistently produce at least 3–8 duplicate pairs.
- `text_dup_pairs` is non-empty in the run report — the dashboard proof section renders.
- `data_reduction_pct` is meaningfully non-zero.
- The dedup threshold is documented in a comment explaining the rationale.

**Todo List**
1. In `agents/deduplicator.py`, change `TEXT_SIM_THRESHOLD = 0.86` to `TEXT_SIM_THRESHOLD = 0.78`.
2. Add a comment explaining: `# 0.78 catches near-duplicate web + academic results from
   overlapping search queries while preserving genuinely distinct entries`.
3. Verify the `_dedup_images()` function: confirm that if `_fetch_image()` fails (timeout),
   the entry is kept but skipped in pairwise comparison — this is the correct behaviour
   and already exists in the code; just verify and add a `print` for failed downloads.

**Relevant Context**
- [`backend/agents/deduplicator.py:31`](backend/agents/deduplicator.py:31) — `TEXT_SIM_THRESHOLD = 0.86`
- [`backend/agents/deduplicator.py:138–143`](backend/agents/deduplicator.py:138) — image hash None-skip logic

**Status:** [ ] pending

---

### Sub-Task 4 — Increase Collector Data Volume

**Intent**
Runs collect only ~30 text entries when the configured limits should yield up to 105
(3 queries × 35 max each). Two causes: DDG text often returns fewer results than `max_results`
for academic topics, and Semantic Scholar silently returns 0 on slow/rate-limited requests.
Fix: raise `S2_MAX` to 25 per query, add a retry for Semantic Scholar on failure,
and add a second DDG text call per query with a slightly broadened search term as a top-up.

**Expected Outcomes**
- Typical runs collect 60–90 text entries (up from 30).
- Semantic Scholar failures are retried once with a 2-second pause before giving up.
- A log line reports per-query counts so failures are visible.
- Image volume: raise `DDG_IMAGE_MAX` to 30 and `WIKIMEDIA_MAX` to 20 to hit 50 images.

**Todo List**
1. In `agents/collector.py`, raise the tuneable constants:
   - `DDG_TEXT_MAX`: 20 → 30
   - `S2_MAX`: 15 → 25
   - `DDG_IMAGE_MAX`: 20 → 30
   - `WIKIMEDIA_MAX`: 15 → 20
2. In `_fetch_semantic_scholar()`, add a single retry on `requests.exceptions.RequestException`
   with a 2-second sleep before the retry attempt.
3. In `collect_data()`, after the main query loop, add a second pass over the 3 queries
   with a slightly broader form (strip words shorter than 4 chars) if the total text count
   is below 40 — a "top-up" pass that runs DDG text only (not S2 again).
4. Add summary `print` at the end of `collect_data()` reporting total raw counts.

**Relevant Context**
- [`backend/agents/collector.py:22–25`](backend/agents/collector.py:22) — tuneable constants block
- [`backend/agents/collector.py:65–97`](backend/agents/collector.py:65) — `_fetch_semantic_scholar()`
- [`backend/agents/collector.py:226–256`](backend/agents/collector.py:226) — `collect_data()` entry point

**Status:** [ ] pending

---

### Sub-Task 5 — Fix Image Dedup (Silent Download Failures)

**Intent**
Image dedup consistently finds 0 pairs because `_fetch_image()` times out on most external
image URLs, leaving `hashes[i] = None` for nearly every entry, and entries with `None` hashes
are skipped in pairwise comparison. Need to: (a) increase timeout, (b) add a fallback to
try the `thumbnail` URL if the main `url` download fails, and (c) add a log count so we
know how many hashes were actually computed vs skipped.

**Expected Outcomes**
- At least 40–60% of images have a successfully computed pHash (vs ~0% currently).
- Image duplicate pairs appear in runs where Wikimedia returns similar images on the same topic.
- A log line reports: `[deduplicator] Image hashes: X computed, Y skipped (download failed)`.

**Todo List**
1. In `agents/deduplicator.py`, in `_fetch_image()`, increase `timeout` from `10` to `15`.
2. In `_dedup_images()`, change the hash computation to try `thumbnail` first, then fall back
   to `url` if thumbnail fails (currently it only tries thumbnail/url once):
   ```python
   img = _fetch_image(thumb_url)
   if img is None and thumb_url != entry.get("url", ""):
       img = _fetch_image(entry.get("url", ""))
   ```
3. After computing all hashes, add a `print` logging how many succeeded vs were `None`.
4. Lower `IMAGE_HASH_THRESHOLD` from `10` to `8` (Hamming distance) to be slightly stricter
   but still catch near-identical scaled/cropped versions.

**Relevant Context**
- [`backend/agents/deduplicator.py:106–113`](backend/agents/deduplicator.py:106) — `_fetch_image()`
- [`backend/agents/deduplicator.py:128–133`](backend/agents/deduplicator.py:128) — hash computation loop
- [`backend/agents/deduplicator.py:103`](backend/agents/deduplicator.py:103) — `IMAGE_HASH_THRESHOLD = 10`

**Status:** [ ] pending

---

## Implementation Order

```
Sub-Task 1 → Sub-Task 2 → Sub-Task 3 → Sub-Task 4 → Sub-Task 5
(frontend UI fix)  (quality filter)  (text dedup)  (volume)   (image dedup)
```

Sub-Task 1 is purely frontend — no backend dependency. Sub-Tasks 2–5 are all backend
and independent of each other (can be reviewed one at a time). Sub-Task 4 feeds more
volume into Sub-Tasks 2, 3, 5 — so after implementing all of them, run one full test
pipeline to verify end-to-end counts are healthy.

## Validation Criteria (run after all sub-tasks complete)
- A test run on "LLM security vulnerabilities" should produce:
  - ≥ 60 raw text entries
  - ≥ 40 entries surviving quality filter
  - ≥ 5 text duplicate pairs found
  - Sample text entries visible in the dashboard (non-blank)
  - Non-zero savings metrics
