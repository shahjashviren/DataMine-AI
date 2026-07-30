# RAG Dataset Engine — Production Rebuild Plan

## Top-Level Overview

**Goal:** Upgrade DataMine AI from a narrow arXiv/Openverse wrapper into a production-grade
Autonomous RAG Dataset Engine. The five agent files and `main.py` are rewritten or extended
in place; no new routes, no new frameworks.

**Scope:**
- `agents/intent_parser.py` — expand 2-keyword output to 3 strategic search queries
- `agents/collector.py` — replace arXiv + Openverse with DuckDuckGo + Semantic Scholar (text) and DuckDuckGo Images + Wikimedia Commons (images); hard-delete all fallback/padding logic
- `agents/cleaner.py` — add HTML tag stripping, citation bracket removal, RAG context chunking (500–1000 chars)
- `agents/structurer.py` — add Alpaca-format instruction synthesis (Groq) as a post-dedup step
- `agents/quality_filter.py` — upgrade text scoring to strict 1–10 Relevance Scale (reject < 7); add Groq Vision image audit (`llama-3.2-11b-vision-preview`)
- `agents/deduplicator.py` — raise similarity threshold to 0.86, verify batched encode is used, tighten metrics
- `backend/requirements.txt` — add `duckduckgo_search`
- `backend/main.py` — thread the 3-query list through the orchestration; add instruction-synthesis SSE step

**Non-goals:** No UI changes, no new API routes, no auth changes, no infrastructure changes.

---

## Sub-Tasks

---

### Sub-Task 1 — Multi-Query Intent Expansion

**Intent**
Replace the current 2-keyword tuple with 3 complementary search queries. This is the root fix
that guarantees semantic overlap exists for the deduplicator to find duplicates in.

**Expected Outcomes**
- `parse_intent()` returns `(topic: str, queries: list[str])` with exactly 3 items in the list.
- Each query is strategically different (different angle on the same topic).
- Fallback path also produces 3 queries.
- All downstream callers (`main.py`, `collector.py`) pass and accept the 3-query list without errors.

**Todo List**
1. Update the Groq system prompt in `intent_parser.py` to request 3 queries (not 2 keywords).
   - Rename the JSON field from `"keywords"` to `"queries"`.
   - Include the Ocean plastic example from the blueprint.
2. Update `parse_intent()` return value: ensure it returns `list[str]` of length 3; pad/trim as needed.
3. Update fallback path to generate 3 slight variations of the raw prompt.
4. Update `main.py` to accept and log all 3 queries in the `intent_done` SSE event.
5. Update `collector.py` `collect_data(topic, keywords)` signature comment (no functional change needed yet — collector loops over `keywords[:2]` today; Sub-Task 2 will change this fully).

**Relevant Context**
- [`backend/agents/intent_parser.py`](backend/agents/intent_parser.py) — full file, 78 lines
- [`backend/main.py`](backend/main.py:116) — `parse_intent` call at line ~116
- Return type change: `tuple[str, list[str]]` stays the same shape; just list length goes 2 → 3

**Status:** [ ] pending

---

### Sub-Task 2 — Live RAG Data Retrieval (Collector Replacement)

**Intent**
Rip out arXiv + Openverse completely and replace with:
- **Text:** DuckDuckGo Search (`DDGS().text()`) + Semantic Scholar API — run against all 3 queries
- **Images:** DuckDuckGo Images (`DDGS().images()`) + Wikimedia Commons — run against topic
- **Zero Padding Rule:** Delete every fallback/default data array; return whatever is actually found

**Expected Outcomes**
- Running a prompt like "Ocean plastic pollution" returns real web + academic results on that topic, not physics papers.
- Running a prompt on biology/law/environment returns domain-relevant content.
- Image results come from DuckDuckGo Images and Wikimedia Commons, not Openverse.
- If a query returns 0 results, the list stays empty — no padding, no defaults.
- 3 overlapping queries produce natural semantic duplicates for the deduplicator.

**Todo List**
1. Add `duckduckgo_search` to `requirements.txt`.
2. In `collector.py`, delete all arXiv and Openverse code (constants, helpers, auth logic).
3. Implement `_fetch_ddg_text(query: str, max_results: int = 20) -> list[dict]` using `DDGS().text()`.
   - Map fields: `title`, `body` → `abstract`, `href` → `id`/`url`, `source = "ddg_web"`.
4. Implement `_fetch_semantic_scholar(query: str, max_results: int = 15) -> list[dict]` using the Semantic Scholar public search API (`https://api.semanticscholar.org/graph/v1/paper/search`).
   - Request fields: `paperId`, `title`, `abstract`, `year`.
   - Skip papers with no abstract.
   - Map to same dict shape as DDG text entries.
5. Implement `_fetch_ddg_images(query: str, max_results: int = 20) -> list[dict]` using `DDGS().images()`.
   - Map fields: `title`, `image` → `url`, `thumbnail`, `source = "ddg_images"`.
6. Implement `_fetch_wikimedia(query: str, max_results: int = 15) -> list[dict]` using Wikimedia Commons API (`https://commons.wikimedia.org/w/api.php` with `action=query&list=search`).
   - Resolve thumbnail URLs from image titles.
   - Map to same image dict shape.
7. Update `collect_data(topic, queries)` to:
   - Loop over all 3 queries for text (DDG + Semantic Scholar each).
   - Run image fetch once for topic (DDG + Wikimedia).
   - Return `(texts, images)` exactly as before — no shape change for downstream.
8. Remove the old `OPENVERSE_CLIENT_ID` / `OPENVERSE_CLIENT_SECRET` env var references.

**Relevant Context**
- [`backend/agents/collector.py`](backend/agents/collector.py) — 284 lines, full replacement
- [`backend/requirements.txt`](backend/requirements.txt) — add `duckduckgo_search`
- Semantic Scholar API is unauthenticated (rate limit: 100 req/5 min); no API key needed
- Wikimedia Commons API is public/unauthenticated
- Dict shape expected by cleaner: `{id, title, abstract, source, search_term}` for text; `{id, title, url, thumbnail, license, tags, source}` for images

**Status:** [ ] pending

---

### Sub-Task 3 — Cleaner: HTML Stripping, Citation Removal & RAG Chunking

**Intent**
Upgrade `cleaner.py` so web-scraped content (which contains HTML, citation brackets, and
long unstructured paragraphs) is properly prepared for instruction synthesis. Add chunking
so long documents become dense 500–1000 character context blocks.

**Expected Outcomes**
- All `<tag>` HTML is stripped from abstracts/bodies.
- Citation patterns like `[1]`, `[12]`, `[Smith et al., 2020]` are removed.
- Existing LaTeX stripping is retained.
- Text longer than 1000 characters is split into chunks; each chunk becomes its own entry with a `chunk_index` field.
- Short chunks (< 500 chars) after splitting are dropped.
- Image cleaning logic is unchanged.

**Todo List**
1. Add HTML stripping via `re.sub(r'<[^>]+>', '', text)` (no new dependencies needed).
2. Add citation removal: `re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)` and `re.sub(r'\[[A-Z][a-z]+.*?\d{4}\]', '', text)`.
3. After cleaning, if `len(abstract) > 1000`, split into overlapping 700-character chunks with 200-character stride, keeping chunks between 500 and 1000 characters.
4. Each chunk entry inherits all parent fields + adds `chunk_index: int` and `chunk_total: int`.
5. Update `clean_data()` to return the expanded (potentially longer) list of chunk entries.
6. Update `_MIN_ABSTRACT_WORDS` threshold check to happen after chunking (apply to each chunk).

**Relevant Context**
- [`backend/agents/cleaner.py`](backend/agents/cleaner.py) — 116 lines
- Downstream consumers: `quality_filter.py` reads `entry["abstract"][:400]` — chunked entries still have `abstract` field, so no change needed downstream
- `structurer.py` reads `entry["abstract"][:200]` for sample display — compatible

**Status:** [ ] pending

---

### Sub-Task 4 — Quality Filter: Strict 1–10 Scoring & Extended Image Heuristics

**Intent**
Replace the binary KEEP/DROP text filter with a scored 1–10 relevance rubric (reject < 7).
Replace the minimal license-only image filter with a rich zero-cost heuristic guard
(keyword relevance on title/tags, domain blacklist, URL pattern checks).
No Groq Vision calls — zero additional API cost.

**Expected Outcomes**
- Every text entry receives a relevance score 1–10; anything below 7 is dropped.
- The system prompt clearly defines the 1–10 scale against the user's original prompt.
- Images are filtered by: topic keyword presence in title/tags, blocked domains (stock sites),
  URL pattern guards (no portraits/logos detected via filename patterns).
- Existing exponential backoff and batch logic is preserved.

**Todo List**
1. In `quality_filter.py`, rewrite `_build_system_prompt()` to instruct Groq to return a score 1–10 per entry (not KEEP/DROP).
   - Format: `1 8` (entry number, space, score).
2. Rewrite `_parse_batch_verdicts()` → `_parse_batch_scores()` to parse integer scores; return `True` if score ≥ 7.
3. Update `_filter_texts()` to use the new score-based parser.
4. Rewrite `_filter_images()` with extended heuristics:
   - Check that at least one topic keyword appears in title or tags (case-insensitive).
   - Block known stock-photo domains: `shutterstock`, `gettyimages`, `istockphoto`, `alamy`, `dreamstime`.
   - Block URL filename patterns: `portrait`, `headshot`, `logo`, `avatar`, `profile-pic`.
   - Retain existing license check as an additional gate.
5. Accept `topic: str` as a new parameter to `_filter_images()` and `filter_quality()`.

**Relevant Context**
- [`backend/agents/quality_filter.py`](backend/agents/quality_filter.py) — 261 lines
- Zero cost: no Vision API calls, no new dependencies
- `filter_quality()` signature change: add `topic` param (already has `user_prompt` and `topic`)

**Status:** [ ] pending

---

### Sub-Task 5 — Deduplicator: Threshold Tuning & Metrics Verification

**Intent**
Raise the cosine similarity threshold from 0.85 → 0.86 as specified. Confirm that batch
encoding (`batch_size=32`) is already in use (it is). Verify that the metrics reported
to the dashboard are derived from the actual duplicate count, not a formula that can
produce zero when queries are orthogonal (this is now fixed by Sub-Task 1's 3-query expansion
which guarantees overlap).

**Expected Outcomes**
- `TEXT_SIM_THRESHOLD` is `0.86`.
- `model.encode()` uses `batch_size=32` (already present — verify and keep).
- `compute_savings()` in `structurer.py` receives the real pre/post counts (already correct — verify).
- No functional regressions in `_dedup_images()`.

**Todo List**
1. Change `TEXT_SIM_THRESHOLD = 0.85` → `0.86` in `deduplicator.py`.
2. Verify `model.encode(abstracts, show_progress_bar=False, batch_size=32)` is present (it is — no change).
3. Verify `_dedup_images()` `kept` logic is correct: the current code has a latent bug where
   `kept_indices.append(i)` is called before the inner loop, then `dropped` items are removed
   at the end — confirm the final list comprehension `[entries[i] for i in kept_indices if i not in dropped]`
   is correct (it is).
4. No changes to `structurer.py` savings formula — it already uses `raw_count - final_count`
   correctly; with 3 overlapping queries, `duplicates_removed` will now be non-zero.

**Relevant Context**
- [`backend/agents/deduplicator.py`](backend/agents/deduplicator.py:30) — `TEXT_SIM_THRESHOLD` on line 30
- [`backend/agents/structurer.py`](backend/agents/structurer.py) — `compute_savings()` function

**Status:** [ ] pending

---

### Sub-Task 6 — Alpaca-Format Instruction Synthesis (Structurer)

**Intent**
After deduplication, pass each surviving text chunk through Groq to synthesize an
Alpaca-format instruction-tuning record. The output JSONL changes from raw abstract dumps
to `{"instruction": "...", "input": "...", "output": "..."}` records suitable for LLM fine-tuning.

**Expected Outcomes**
- `structure_output()` calls a new `_synthesize_instructions()` helper before writing the JSONL.
- Each text entry is converted to an Alpaca record; entries where synthesis fails retain raw format.
- The output `texts.jsonl` contains Alpaca-format JSON lines.
- `run_report.json` `sample_texts` shows Alpaca fields, not raw abstracts.
- A new `instruction_synthesis` SSE step is emitted from `main.py` between `dedup_done` and `structure`.

**Todo List**
1. In `structurer.py`, add `_synthesize_instructions(entries: list[dict], topic: str) -> list[dict]`:
   - Batch entries in groups of 10 (Groq context is smaller for generation tasks).
   - System prompt: instructs Groq (`llama-3.3-70b-versatile`) to return a JSON array of
     `{"instruction": "...", "input": "...", "output": "..."}` records from the chunk texts.
   - On parse failure for a batch, fall back to a simple template:
     `{"instruction": "Describe: " + title, "input": abstract[:500], "output": abstract[500:1000]}`.
   - Return the list of Alpaca dicts.
2. In `structure_output()`, call `_synthesize_instructions(final_texts, topic)` before
   calling `_write_text_jsonl()`.
3. Update `_write_text_jsonl()` to write the synthesized Alpaca records (not the raw entries).
4. Update `sample_texts` in the report to show `instruction` and `output` fields.
5. In `main.py`, add an `instruction_synthesis` SSE step between `dedup_done` and `structure`:
   - Emit: `"Synthesizing Alpaca instruction-tuning records via Groq…"`
   - This step is handled inside `structure_output()` — the SSE event wraps the whole call.
   - Update the existing `structure` emit message to reflect both synthesis + file writing.

**Relevant Context**
- [`backend/agents/structurer.py`](backend/agents/structurer.py) — `structure_output()` entry point
- [`backend/main.py`](backend/main.py:190) — Step 6 structure block
- Groq model for synthesis: `llama-3.3-70b-versatile` (same as intent parser)
- The synthesis step is the slowest new addition; keepalive pings via `run_in_executor` pattern already used in `main.py` should be applied here too

**Status:** [ ] pending

---

### Sub-Task 7 — Dependency & Environment Cleanup

**Intent**
Ensure `requirements.txt` reflects the new sources, remove unused Openverse env vars
from `.env.example`, and confirm no import regressions.

**Expected Outcomes**
- `duckduckgo_search` is listed in `requirements.txt`.
- `OPENVERSE_CLIENT_ID` and `OPENVERSE_CLIENT_SECRET` are removed from `.env.example`.
- No unused imports remain in any agent file.
- `.env.example` has a comment noting Semantic Scholar and Wikimedia require no API keys.

**Todo List**
1. Add `duckduckgo-search` to `requirements.txt`.
2. Open `backend/.env.example`, remove `OPENVERSE_CLIENT_ID` and `OPENVERSE_CLIENT_SECRET` lines.
3. Add a comment: `# Semantic Scholar and Wikimedia Commons APIs require no key`.
4. Remove any lingering `import requests` usage in `collector.py` that only served arXiv/Openverse (keep it if Semantic Scholar or Wikimedia use it).
5. Verify all `from agents.X import Y` in `main.py` still resolve after refactors.

**Relevant Context**
- [`backend/requirements.txt`](backend/requirements.txt)
- [`backend/.env.example`](backend/.env.example)

**Status:** [ ] pending

---

## Implementation Order

```
Sub-Task 1 → Sub-Task 2 → Sub-Task 3 → Sub-Task 4 → Sub-Task 5 → Sub-Task 6 → Sub-Task 7
(intent)      (collector)   (cleaner)    (quality)     (dedup)      (structurer)  (deps)
```

Each sub-task is independent of later ones but feeds into them via shared data shapes. They
must be executed in order because each agent receives the output of the previous.
