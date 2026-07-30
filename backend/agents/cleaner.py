"""
Cleaner agent.

Cleans each raw text entry in-place and returns at most one output entry per
input entry (1-in / 1-out).  The entry count can only stay the same or
decrease (boilerplate/empty entries are dropped).  It never increases.

Operations applied per entry:
  - Strip HTML tags
  - Remove citation brackets [1], [1,2], [Smith et al., 2020]
  - Strip LaTeX math / command syntax
  - Unicode NFKC normalisation + whitespace collapse
  - Truncate to _MAX_ABSTRACT_CHARS (keeps abstracts a sane length)
  - Drop entries that are entirely boilerplate or empty after cleaning

Chunking has been removed.  Splitting one document into N overlapping
windows inflated the "cleaned" count shown to the user and caused the
deduplicator to compare fragments of the same article.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS = [
    re.compile(r"abstract\s+not\s+available", re.IGNORECASE),
    re.compile(r"no\s+abstract", re.IGNORECASE),
    re.compile(r"^\s*n/?a\s*$", re.IGNORECASE),
]

_MIN_ABSTRACT_CHARS = 80    # drop entries shorter than this after cleaning
_MAX_ABSTRACT_CHARS = 3000  # truncate very long text to a useful reading length


def _strip_html(text: str) -> str:
    """Remove all HTML/XML tags."""
    return re.sub(r"<[^>]+>", " ", text)


def _strip_citations(text: str) -> str:
    """Remove inline citation brackets: [1], [1,2], [Smith et al., 2020]."""
    text = re.sub(r"\[\d+(?:[,\s]+\d+)*\]", "", text)
    text = re.sub(r"\[[A-Z][a-zA-Z\s]+et\s+al\.,?\s+\d{4}\]", "", text)
    text = re.sub(r"\[[A-Z][a-zA-Z]+,?\s+\d{4}\]", "", text)
    return text


def _strip_latex(text: str) -> str:
    """Remove LaTeX math and command syntax."""
    text = re.sub(r"\$\$[^$]*\$\$", "", text)          # display math
    text = re.sub(r"\$[^$]*\$", "", text)               # inline math
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", text)   # \cmd{...}
    text = re.sub(r"\\[a-zA-Z]+", "", text)             # bare \cmd
    return text


def _normalize(text: str) -> str:
    """Unicode normalization + whitespace collapse."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_raw_text(text: str) -> str:
    """Apply the full cleaning pipeline to a raw text string."""
    text = _strip_html(text)
    text = _strip_citations(text)
    text = _strip_latex(text)
    text = _normalize(text)
    return text[:_MAX_ABSTRACT_CHARS]


def _clean_text_entry(entry: dict) -> dict | None:
    """
    Clean a single text entry and return it, or None if it should be dropped.

    One entry in, at most one entry out — no splitting/chunking.
    parent_document_id is set to the entry's own id/url so the deduplicator
    can still distinguish cross-document pairs correctly.
    """
    title    = entry.get("title", "").strip()
    abstract = entry.get("abstract", "").strip()

    if not title or not abstract:
        return None

    title    = _normalize(_strip_html(title))
    abstract = _clean_raw_text(abstract)

    # Drop boilerplate
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(abstract):
            return None

    # Drop entries that are too short to be useful after cleaning
    if len(abstract) < _MIN_ABSTRACT_CHARS:
        return None

    # parent_document_id = the entry's own stable identifier.
    # Used by the deduplicator to skip self-comparisons if the same URL ever
    # appears twice, and to correctly label dup-pair entries in the report.
    parent_id = entry.get("id") or entry.get("url") or f"{entry.get('source', '')}:{title}"

    return {
        **entry,
        "title":              title,
        "abstract":           abstract,
        "parent_document_id": parent_id,
    }


# ---------------------------------------------------------------------------
# Image cleaning helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def _clean_image_entry(entry: dict) -> dict | None:
    url       = entry.get("url", "").strip()
    thumbnail = entry.get("thumbnail", "").strip()

    if not url and not thumbnail:
        return None

    title = unicodedata.normalize("NFKC", entry.get("title", "")).strip()
    title = re.sub(r"\s+", " ", title)

    return {**entry, "title": title, "url": url, "thumbnail": thumbnail}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def clean_data(
    raw_texts: list[dict], raw_images: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Clean raw text and image records.
    Returns (clean_texts, clean_images).

    INVARIANT: len(clean_texts) <= len(raw_texts).
    Each raw entry produces exactly one cleaned entry or is dropped.
    No splitting or chunking — the count can only stay the same or decrease.
    """
    clean_texts: list[dict] = []
    for entry in raw_texts:
        cleaned = _clean_text_entry(entry)
        if cleaned is not None:
            clean_texts.append(cleaned)

    dropped = len(raw_texts) - len(clean_texts)
    print(
        f"[cleaner] {len(raw_texts)} raw text entries -> {len(clean_texts)} clean "
        f"({dropped} dropped as boilerplate/too-short/empty)."
    )

    # Hard invariant: cleaned count must never exceed raw count.
    if len(clean_texts) > len(raw_texts):
        raise RuntimeError(
            f"[cleaner] INVARIANT VIOLATED: {len(clean_texts)} clean > "
            f"{len(raw_texts)} raw. Chunking must have been re-introduced."
        )

    clean_images: list[dict] = []
    for entry in raw_images:
        cleaned_img = _clean_image_entry(entry)
        if cleaned_img is not None:
            clean_images.append(cleaned_img)

    print(f"[cleaner] {len(raw_images)} images -> {len(clean_images)} clean images.")
    return clean_texts, clean_images
