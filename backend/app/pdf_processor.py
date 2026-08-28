"""
PDF Processor, extract text from PDF files using PyMuPDF.

Raw `page.get_text()` returns everything the page draws, including furniture that is
not part of the document's prose: LaTeX line numbers, running headers, page numbers.
That furniture is poison for RAG. It occupies chunk budget, it embeds to nothing
meaningful, and the generator echoes it into answers ("...the research. 401 Answer:").
On a line-numbered conference submission it can exceed half of all extracted lines, so
cleaning it is not cosmetic; it decides whether retrieval works at all.
"""

import re
from collections import Counter

import pymupdf

# A line holding nothing but 1-4 digits. Could be a line number, a page number, or a
# lone table cell, so matching this is necessary but never sufficient to drop it.
_BARE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# Below this many bare numbers there is no sequence to reason about, and stripping
# would be guesswork against real content.
_MIN_NUMBERS_TO_JUDGE = 10

# Share of adjacent bare numbers that must increment by exactly 1 for the document to
# count as line-numbered. Table columns do not count up one at a time down the page.
_INCREMENT_RATIO = 0.8

# A line must appear on at least this share of pages to read as running furniture.
_HEADER_PAGE_RATIO = 0.6
_HEADER_MIN_PAGES = 3
_HEADER_MAX_CHARS = 120


def _is_line_numbered(values: list[int]) -> bool:
    """True if these bare numbers form a counter rather than data.

    Judged on the sequence, not on position, so it holds whether the extractor emits
    numbers in a block (common in two-column layouts) or interleaved with text lines.
    """
    if len(values) < _MIN_NUMBERS_TO_JUDGE:
        return False
    increments = sum(1 for a, b in zip(values, values[1:]) if b == a + 1)
    return increments >= _INCREMENT_RATIO * (len(values) - 1)


def _strip_line_numbers(text: str) -> str:
    """Drop LaTeX line-number artifacts, but only from documents that actually have them."""
    lines = text.split("\n")
    numbered = [(i, int(l.strip())) for i, l in enumerate(lines) if _BARE_NUMBER_RE.match(l)]
    if not _is_line_numbered([v for _, v in numbered]):
        return text
    drop = {i for i, _ in numbered}
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


def _find_running_headers(page_texts: list[str]) -> set[str]:
    """Lines repeated across most pages: venue headers, footers, page furniture.

    Counted once per page, so a phrase used several times on a single page is not
    mistaken for a header. Long lines are exempt, since repeated prose is real content.
    """
    if len(page_texts) < _HEADER_MIN_PAGES:
        return set()
    seen = Counter()
    for page in page_texts:
        for line in {l.strip() for l in page.split("\n") if l.strip()}:
            if len(line) <= _HEADER_MAX_CHARS:
                seen[line] += 1
    threshold = max(_HEADER_MIN_PAGES, _HEADER_PAGE_RATIO * len(page_texts))
    return {line for line, count in seen.items() if count >= threshold}


def _clean_page(text: str, headers: set[str]) -> str:
    kept = [l for l in text.split("\n") if l.strip() not in headers]
    return _strip_line_numbers("\n".join(kept))


def extract_text_from_pdf(pdf_path: str) -> tuple[str, list[str]]:
    """
    Extract text from a PDF file, removing page furniture that would otherwise be
    chunked, embedded, and quoted back to the user as though it were content.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Tuple of (full_text, list_of_page_texts)
    """
    doc = pymupdf.open(pdf_path)
    try:
        raw_pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()

    headers = _find_running_headers(raw_pages)
    page_texts = [_clean_page(text, headers) for text in raw_pages]

    # Collapse the blank runs that removing furniture leaves behind, so chunkers that
    # split on paragraph breaks do not emit a pile of empty chunks.
    full_text = re.sub(r"\n{3,}", "\n\n", "\n\n".join(page_texts))
    return full_text, page_texts
