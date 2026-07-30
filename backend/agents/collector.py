"""
Collector agent — Multi-source RAG edition.

Text sources (in priority order):
  1. Wikipedia REST API     — encyclopedic summaries via search+summary endpoints
  2. Semantic Scholar API   — academic papers, queried with TWO complementary terms
                              to guarantee overlapping results for the deduplicator

  arXiv has been removed entirely.
  DDG web is kept as a thin supplement when the two primary sources are lean.

Images: Openverse API (primary, CC-licensed) + Wikimedia Commons + DuckDuckGo Images.

IMPORTANT: both Wikipedia and Semantic Scholar (and Wikimedia Commons) require a
custom User-Agent header on every request; without it Wikipedia returns 403.
All requests in this module include _HEADERS.
"""

import os
import re
import time
import requests
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

S2_MAX           = 30   # Semantic Scholar results per query term
WIKI_MAX         = 10   # Wikipedia search results per query
DDG_TEXT_MAX     = 25   # DDG web results per query (supplement only)
DDG_IMAGE_MAX    = 20   # DDG image results per image query
WIKIMEDIA_MAX    = 15   # Wikimedia Commons image results per query
OPENVERSE_MAX    = 20   # Openverse results per image query

# If primary sources return fewer than this per query, supplement with DDG
MIN_TEXT_BEFORE_DDG = 10

# Minimum total text entries before pipeline proceeds (top-up if below this)
MIN_TOTAL_TEXT = 50

S2_API        = "https://api.semanticscholar.org/graph/v1/paper/search"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIPEDIA_SEARCH_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_REST_API   = "https://en.wikipedia.org/api/rest_v1/page/summary/"
OPENVERSE_API = "https://api.openverse.org/v1/images/"

# Polite delay between outbound requests (seconds)
_REQUEST_DELAY = 0.8

# ---------------------------------------------------------------------------
# Shared User-Agent header — required by Wikipedia and Wikimedia Commons.
# Without a descriptive User-Agent, Wikipedia returns 403 / rate-limits.
# Semantic Scholar tolerates missing UA but it is polite to send one.
# ---------------------------------------------------------------------------
_UA = "DataMineAI/2.0 (https://github.com/datamine-ai; open-source research pipeline)"
_HEADERS = {"User-Agent": _UA}


# ---------------------------------------------------------------------------
# Relevance guard — keyword filter for DDG results
# ---------------------------------------------------------------------------

def _extract_topic_keywords(topic: str, query: str) -> list[str]:
    """
    Extract significant keywords from the topic and query strings.
    Returns lowercase words longer than 3 chars, deduplicated.
    Used to hard-filter DDG results for topical relevance.
    """
    combined = f"{topic} {query}"
    words = re.split(r"\W+", combined.lower())
    stop = {
        "need", "data", "about", "with", "from", "that", "this", "have",
        "what", "which", "when", "where", "will", "more", "into", "some",
        "them", "then", "than", "there", "their", "these", "training",
        "research", "study", "studies", "using", "based", "used", "both",
    }
    seen = set()
    keywords = []
    for w in words:
        if len(w) > 3 and w not in stop and w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords


def _is_topically_relevant(title: str, body: str, keywords: list[str]) -> bool:
    """
    Returns True if at least 1 topic keyword appears in the title+body combined.
    If keywords list is empty, passes everything through.
    """
    if not keywords:
        return True
    haystack = f"{title} {body}".lower()
    return any(kw in haystack for kw in keywords)


# ---------------------------------------------------------------------------
# Semantic Scholar — queried with TWO complementary search terms per topic
# ---------------------------------------------------------------------------

def _fetch_semantic_scholar(query: str, max_results: int = S2_MAX) -> list[dict]:
    """
    Academic papers from Semantic Scholar (no API key required).
    Returns only papers with a non-empty abstract.
    Retries once on failure with a 2-second pause.

    User-Agent header is included as a courtesy; S2 does not require it but
    Wikipedia/Wikimedia do (same _HEADERS dict used across all sources).
    """
    entries: list[dict] = []
    params = {
        "query":  query,
        "fields": "paperId,title,abstract,year,authors",
        "limit":  max_results,
    }
    for attempt in range(1, 3):
        try:
            resp = requests.get(S2_API, params=params, headers=_HEADERS, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            for paper in data.get("data", []):
                title    = (paper.get("title") or "").strip()
                abstract = (paper.get("abstract") or "").strip()
                pid      = paper.get("paperId", "")
                if not title or not abstract:
                    continue
                entries.append({
                    "id":          f"s2:{pid}",
                    "title":       title,
                    "abstract":    abstract,
                    "year":        paper.get("year"),
                    "url":         f"https://www.semanticscholar.org/paper/{pid}",
                    "source":      "semantic_scholar",
                    "search_term": query,
                })
            print(f"[collector] Semantic Scholar '{query[:55]}': {len(entries)} papers")
            break
        except Exception as exc:
            print(f"[collector] Semantic Scholar attempt {attempt}/2 failed for '{query}': {exc}")
            if attempt < 2:
                time.sleep(2.0)
    return entries


# ---------------------------------------------------------------------------
# Wikipedia — search API to find pages, then REST summary for each page
# ---------------------------------------------------------------------------

def _fetch_wikipedia(query: str, max_results: int = WIKI_MAX) -> list[dict]:
    """
    Encyclopedic summaries from Wikipedia.

    Step 1 — action=query&list=search to get the top page titles for the query.
    Step 2 — GET /api/rest_v1/page/summary/{title} for each page to retrieve
             a clean plain-text extract (the REST summary endpoint returns the
             lead section without HTML, typically 3-10 sentences).

    The custom User-Agent header is MANDATORY for Wikipedia to return 200.
    Without it Wikipedia returns 403 Forbidden (enforced since 2023).
    """
    entries: list[dict] = []
    try:
        # Step 1: search for matching page titles
        search_params = {
            "action":   "query",
            "list":     "search",
            "srsearch": query,
            "srlimit":  max_results,
            "format":   "json",
        }
        resp = requests.get(
            WIKIPEDIA_SEARCH_API, params=search_params,
            headers=_HEADERS, timeout=20
        )
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            print(f"[collector] Wikipedia search '{query[:55]}': 0 results")
            return []

        # Step 2: fetch REST summary for each page title
        fetched = 0
        for hit in results:
            page_title = hit.get("title", "").strip()
            if not page_title:
                continue
            # URL-encode spaces as underscores for the REST endpoint
            safe_title = page_title.replace(" ", "_")
            try:
                sum_resp = requests.get(
                    f"{WIKIPEDIA_REST_API}{safe_title}",
                    headers=_HEADERS, timeout=15,
                    params={"redirect": "true"},
                )
                if sum_resp.status_code == 404:
                    continue
                sum_resp.raise_for_status()
                page = sum_resp.json()
                title   = (page.get("title") or page_title).strip()
                extract = (page.get("extract") or "").strip()
                page_id = page.get("pageid", "")
                # Require a meaningful extract (at least 2 sentences / ~150 chars)
                if not extract or len(extract) < 80:
                    continue
                entries.append({
                    "id":          f"wiki:{page_id}",
                    "title":       title,
                    "abstract":    extract,
                    "url":         f"https://en.wikipedia.org/wiki/{safe_title}",
                    "source":      "wikipedia",
                    "search_term": query,
                })
                fetched += 1
            except Exception as page_exc:
                print(f"[collector] Wikipedia summary failed for '{page_title}': {page_exc}")
                continue
            time.sleep(0.15)   # polite inter-page delay

        print(f"[collector] Wikipedia '{query[:55]}': {fetched}/{len(results)} pages fetched")
    except Exception as exc:
        print(f"[collector] Wikipedia search failed for '{query}': {exc}")
    return entries


# ---------------------------------------------------------------------------
# DuckDuckGo text (supplement only — keyword-filtered)
# ---------------------------------------------------------------------------

def _fetch_ddg_text(
    query: str, keywords: list[str], max_results: int = DDG_TEXT_MAX
) -> list[dict]:
    """
    Web search via DuckDuckGo. Used as a supplement when primary sources are thin.
    HARD FILTERS every result: title+body must contain at least one topic keyword.
    """
    entries: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                title    = (r.get("title") or "").strip()
                abstract = (r.get("body") or "").strip()
                url      = (r.get("href") or "").strip()
                if not title or not abstract:
                    continue
                if not _is_topically_relevant(title, abstract, keywords):
                    continue
                entries.append({
                    "id":          url,
                    "title":       title,
                    "abstract":    abstract,
                    "url":         url,
                    "source":      "ddg_web",
                    "search_term": query,
                })
    except Exception as exc:
        print(f"[collector] DDG text failed for '{query}': {exc}")
    return entries


# ---------------------------------------------------------------------------
# Image fetchers — DDG Images, Wikimedia Commons, Openverse
# ---------------------------------------------------------------------------

def _ddg_images_raw(query: str, max_results: int) -> list[dict]:
    """Inner DDG images call — returns mapped entries, empty list on any failure."""
    entries: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=max_results):
                url       = (r.get("image") or "").strip()
                thumbnail = (r.get("thumbnail") or url).strip()
                title     = (r.get("title") or "").strip()
                if not url:
                    continue
                entries.append({
                    "id":        url,
                    "title":     title,
                    "url":       url,
                    "thumbnail": thumbnail,
                    "license":   "unknown",
                    "tags":      [],
                    "source":    "ddg_images",
                })
    except Exception as exc:
        print(f"[collector] DDG images error for '{query}': {exc}")
    return entries


def _fetch_ddg_images(query: str, max_results: int = DDG_IMAGE_MAX) -> list[dict]:
    """DDG image search. Falls back to 2-word broad query if zero results."""
    entries = _ddg_images_raw(query, max_results)
    if entries:
        return entries
    words = [w for w in query.split() if len(w) > 2]
    if len(words) >= 2:
        broad = " ".join(words[:2])
        if broad.lower() != query.lower():
            print(f"[collector] DDG images: 0 for '{query}' — retrying with '{broad}'")
            return _ddg_images_raw(broad, max_results)
    return []


def _fetch_openverse(query: str, max_results: int = OPENVERSE_MAX) -> list[dict]:
    """
    Fetch CC-licensed images from Openverse (api.openverse.org).
    Logs the request URL, HTTP status, result count, and any errors explicitly.
    """
    entries: list[dict] = []
    params = {
        "q":            query,
        "page_size":    max_results,
        "license_type": "commercial,modification",
        "format":       "json",
    }
    client_id     = os.environ.get("OPENVERSE_CLIENT_ID", "")
    client_secret = os.environ.get("OPENVERSE_CLIENT_SECRET", "")
    headers       = dict(_HEADERS)   # copy; add auth below if credentials present
    token: str    = ""

    if client_id and client_secret:
        try:
            token_resp = requests.post(
                "https://api.openverse.org/v1/auth_tokens/token/",
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     client_id,
                    "client_secret": client_secret,
                },
                headers=headers,
                timeout=15,
            )
            if token_resp.status_code == 200:
                token = token_resp.json().get("access_token", "")
                print("[collector] Openverse: obtained bearer token.")
            else:
                print(
                    f"[collector] Openverse: token request HTTP {token_resp.status_code}"
                    " — falling back to anonymous access."
                )
        except Exception as exc:
            print(f"[collector] Openverse: token request failed ({exc}) — anonymous.")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    print(f"[collector] Openverse: GET {OPENVERSE_API} q='{query}'")
    try:
        resp = requests.get(OPENVERSE_API, params=params, headers=headers, timeout=20)
        print(f"[collector] Openverse: HTTP {resp.status_code} for query='{query}'")
        if resp.status_code != 200:
            print(f"[collector] Openverse: error body — {resp.text[:400]}")
            return []
        data    = resp.json()
        results = data.get("results", [])
        print(
            f"[collector] Openverse: {len(results)} results "
            f"(total={data.get('count', '?')}) for query='{query}'"
        )
        for item in results:
            url       = (item.get("url") or "").strip()
            thumbnail = (item.get("thumbnail") or url).strip()
            title     = (item.get("title") or "").strip()
            license_  = (item.get("license") or "unknown").lower()
            tags      = [t.get("name", "") for t in item.get("tags", []) if t.get("name")]
            if not url:
                continue
            entries.append({
                "id":        item.get("id", url),
                "title":     title,
                "url":       url,
                "thumbnail": thumbnail,
                "license":   license_,
                "tags":      tags,
                "source":    "openverse",
            })
        print(f"[collector] Openverse: {len(entries)} entries saved for query='{query}'")
    except Exception as exc:
        print(f"[collector] Openverse: exception for query='{query}': {exc}")
    return entries


def _fetch_wikimedia(query: str, max_results: int = WIKIMEDIA_MAX) -> list[dict]:
    """
    Open-license images from Wikimedia Commons.
    User-Agent header is mandatory — Wikimedia enforces it just like Wikipedia.
    """
    entries: list[dict] = []
    try:
        search_params = {
            "action":      "query",
            "list":        "search",
            "srsearch":    f"{query} filetype:bitmap",
            "srnamespace": 6,
            "srlimit":     max_results,
            "format":      "json",
        }
        resp = requests.get(
            WIKIMEDIA_API, params=search_params,
            headers=_HEADERS, timeout=20
        )
        resp.raise_for_status()
        titles = [
            hit["title"]
            for hit in resp.json().get("query", {}).get("search", [])
        ]
        if not titles:
            return []
        info_params = {
            "action":     "query",
            "titles":     "|".join(titles),
            "prop":       "imageinfo",
            "iiprop":     "url|mime",
            "iiurlwidth": 400,
            "format":     "json",
        }
        info_resp = requests.get(
            WIKIMEDIA_API, params=info_params,
            headers=_HEADERS, timeout=20
        )
        info_resp.raise_for_status()
        pages = info_resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            title     = page.get("title", "")
            info_list = page.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]
            if not info.get("mime", "").startswith("image/"):
                continue
            url       = info.get("url", "")
            thumbnail = info.get("thumburl") or url
            if not url:
                continue
            entries.append({
                "id":        url,
                "title":     title.replace("File:", "").strip(),
                "url":       url,
                "thumbnail": thumbnail,
                "license":   "cc-by-sa",
                "tags":      [],
                "source":    "wikimedia",
            })
    except Exception as exc:
        print(f"[collector] Wikimedia failed for '{query}': {exc}")
    return entries


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def collect_data(topic: str, queries: list[str]) -> tuple[list[dict], list[dict]]:
    """
    Collect raw text and image entries from authoritative sources.

    Text strategy:
      For each of the 3 intent queries:
        1. Wikipedia REST (search + per-page summary) — always run
        2. Semantic Scholar query A  = the intent query itself
        3. Semantic Scholar query B  = a complementary rephrasing of the query
           (generated by prepending topic keywords to guarantee overlap)
        4. DDG web — only if 1+2+3 together yield < MIN_TEXT_BEFORE_DDG entries
                     AND every DDG result is hard-filtered by topic keywords.

    Two S2 queries per topic intentionally produce overlapping papers so the
    deduplicator always has genuine near-duplicate pairs to flag.

    Image strategy:
      Openverse (primary) + Wikimedia Commons + DDG Images.

    Returns (texts, images).
    """
    texts: list[dict] = []
    seen_text_ids: set[str] = set()
    all_keywords = _extract_topic_keywords(topic, " ".join(queries))

    def _add_texts(new_entries: list[dict]) -> int:
        added = 0
        for e in new_entries:
            uid = e.get("id") or e.get("url") or ""
            if uid not in seen_text_ids:
                texts.append(e)
                seen_text_ids.add(uid)
                added += 1
        return added

    def _make_s2_query_b(query: str, topic: str) -> str:
        """
        Produce a complementary Semantic Scholar query from the primary query.
        Strategy: take the topic keywords and prefix with the first content word
        of the primary query that isn't already in the topic, creating a
        semantically related but differently-phrased search that will overlap.
        """
        topic_words = set(re.split(r"\W+", topic.lower()))
        query_words = [w for w in re.split(r"\W+", query.lower()) if len(w) > 3]
        # Add words from the query that aren't in the topic to diversify
        extra = [w for w in query_words if w not in topic_words]
        base  = topic
        if extra:
            base = f"{topic} {extra[0]}"
        # Append a different framing suffix to create a distinct but overlapping query
        suffixes = ["overview", "analysis", "systems", "methods", "impact"]
        # Pick suffix based on hash of the query for determinism across runs
        suffix = suffixes[hash(query) % len(suffixes)]
        return f"{base} {suffix}"

    for query in queries:
        query_keywords = _extract_topic_keywords(topic, query)

        # Primary sources: Wikipedia + two S2 queries
        wiki_results = _fetch_wikipedia(query)
        time.sleep(_REQUEST_DELAY)

        s2_query_a = query
        s2_query_b = _make_s2_query_b(query, topic)
        s2a_results = _fetch_semantic_scholar(s2_query_a)
        time.sleep(_REQUEST_DELAY)
        s2b_results = _fetch_semantic_scholar(s2_query_b)
        time.sleep(_REQUEST_DELAY)

        primary_count = (
            _add_texts(wiki_results)
            + _add_texts(s2a_results)
            + _add_texts(s2b_results)
        )
        print(
            f"[collector] '{query[:55]}': "
            f"{len(wiki_results)} Wiki + {len(s2a_results)} S2-A + {len(s2b_results)} S2-B"
            f" = {primary_count} new"
        )

        # DDG supplement when primary sources are thin
        if primary_count < MIN_TEXT_BEFORE_DDG:
            ddg_results = _fetch_ddg_text(query, query_keywords)
            added_ddg   = _add_texts(ddg_results)
            print(f"[collector]   + {added_ddg} DDG web (supplement, keyword-filtered)")
            time.sleep(_REQUEST_DELAY)

    # Top-up: if we still don't have MIN_TOTAL_TEXT, run DDG on ALL queries
    if len(texts) < MIN_TOTAL_TEXT:
        print(
            f"[collector] Only {len(texts)} entries — top-up DDG pass to reach {MIN_TOTAL_TEXT}..."
        )
        for query in queries:
            if len(texts) >= MIN_TOTAL_TEXT:
                break
            query_keywords = _extract_topic_keywords(topic, query)
            ddg_extra = _fetch_ddg_text(query, query_keywords, max_results=DDG_TEXT_MAX)
            added = _add_texts(ddg_extra)
            print(f"[collector] top-up '{query[:50]}': +{added} new entries")
            time.sleep(_REQUEST_DELAY)

    print(f"[collector] total text collected: {len(texts)}")

    # ---------------------------------------------------------------------------
    # Images — Openverse + Wikimedia Commons + DDG Images
    # ---------------------------------------------------------------------------
    MIN_IMAGES = 10

    img_keywords = _extract_topic_keywords(topic, " ".join(queries))

    def _img_url_relevant(img: dict) -> bool:
        if img.get("source") in ("wikimedia", "openverse"):
            return True
        if not img_keywords:
            return True
        haystack = (img.get("url", "") + " " + img.get("title", "")).lower()
        return any(kw in haystack for kw in img_keywords)

    def _collect_images_for_query(iq: str, seen: set) -> tuple[list, list, list]:
        """Returns (openverse_imgs, wiki_imgs, ddg_imgs) for one query."""
        ov_imgs: list[dict] = []
        wiki: list[dict] = []
        ddg: list[dict] = []
        for img in _fetch_openverse(iq, max_results=OPENVERSE_MAX):
            u = img.get("url", "")
            if u and u not in seen:
                ov_imgs.append(img)
                seen.add(u)
        for img in _fetch_wikimedia(iq, max_results=WIKIMEDIA_MAX):
            u = img.get("url", "")
            if u and u not in seen:
                wiki.append(img)
                seen.add(u)
        for img in _fetch_ddg_images(iq, max_results=DDG_IMAGE_MAX):
            u = img.get("url", "")
            if u and u not in seen and _img_url_relevant(img):
                ddg.append(img)
                seen.add(u)
        return ov_imgs, wiki, ddg

    image_queries  = list(dict.fromkeys([topic] + queries))
    openverse_images: list[dict] = []
    wiki_images: list[dict] = []
    ddg_images: list[dict] = []
    seen_img_urls: set[str] = set()

    for iq in image_queries:
        ov, w, d = _collect_images_for_query(iq, seen_img_urls)
        openverse_images.extend(ov)
        wiki_images.extend(w)
        ddg_images.extend(d)
        time.sleep(_REQUEST_DELAY)

    total_images = len(openverse_images) + len(wiki_images) + len(ddg_images)
    print(
        f"[collector] images round 1: {len(openverse_images)} Openverse + "
        f"{len(wiki_images)} Wikimedia + {len(ddg_images)} DDG = {total_images}"
    )

    # Top-up if below minimum
    if total_images < MIN_IMAGES:
        print(f"[collector] Images below {MIN_IMAGES} — running top-up queries...")
        topup_terms: list[str] = []
        for q in queries:
            words = [w for w in q.split() if len(w) > 3]
            if len(words) >= 2:
                topup_terms.append(f"{words[0]} {words[1]}")
            elif words:
                topup_terms.append(words[0])
        topup_terms = list(dict.fromkeys(topup_terms))

        for term in topup_terms:
            if total_images >= MIN_IMAGES:
                break
            ov, w, d = _collect_images_for_query(term, seen_img_urls)
            openverse_images.extend(ov)
            wiki_images.extend(w)
            ddg_images.extend(d)
            total_images = len(openverse_images) + len(wiki_images) + len(ddg_images)
            print(f"[collector] top-up '{term}': now {total_images} images")
            time.sleep(_REQUEST_DELAY)

    images = openverse_images + wiki_images + ddg_images
    print(
        f"[collector] images final: {len(openverse_images)} Openverse + "
        f"{len(wiki_images)} Wikimedia + {len(ddg_images)} DDG = {len(images)}"
    )
    return texts, images
