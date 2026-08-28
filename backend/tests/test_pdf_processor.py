"""PDF text cleaning: line-number and running-header removal.

Regression origin: uploading a line-numbered conference submission produced chunks that
were more than half bare integers, and the generator quoted them back into answers
("...the reproducibility of the research. 401 Answer: [NA] 402").
"""
import pytest

from app import pdf_processor as pp


def _numbered_page(start, body_lines):
    """A page as PyMuPDF emits it for LaTeX \\linenumbers: numbers, then prose."""
    nums = "\n".join(f"{start + i:03d}" for i in range(len(body_lines)))
    return nums + "\n" + "\n".join(body_lines)


def test_strips_line_numbers_from_numbered_document():
    page = _numbered_page(0, [f"Sentence number {i} of the paper." for i in range(15)])
    cleaned = pp._strip_line_numbers(page)
    assert "000" not in cleaned and "014" not in cleaned
    assert "Sentence number 7 of the paper." in cleaned


def test_keeps_bare_numbers_that_are_not_a_counter():
    """Table cells and stray figures must survive: only a counting sequence is furniture."""
    table = "Iron\n18\nZinc\n11\nCalcium\n1000\nSodium\n2300"
    assert pp._strip_line_numbers(table) == table


def test_keeps_short_number_runs():
    """Too few bare numbers to establish a sequence, so nothing is dropped."""
    text = "Chapter\n1\n2\n3\nBody text here."
    assert pp._strip_line_numbers(text) == text


@pytest.mark.parametrize("values,expected", [
    (list(range(0, 40)), True),                      # clean counter
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99, 11, 12], True),   # counter with one page break
    ([18, 11, 1000, 2300, 5, 60, 700, 2, 9, 44], False),   # table data
    ([1, 2, 3], False),                              # too short to judge
])
def test_line_number_detection(values, expected):
    assert pp._is_line_numbered(values) is expected


def test_finds_running_headers_across_pages():
    pages = [f"Under review at VENUE\nUnique body {i}\nmore text" for i in range(6)]
    headers = pp._find_running_headers(pages)
    assert "Under review at VENUE" in headers
    assert not any(h.startswith("Unique body") for h in headers)


def test_running_header_needs_enough_pages():
    """Two pages cannot establish a pattern, so nothing is treated as furniture."""
    assert pp._find_running_headers(["Header\nbody", "Header\nbody"]) == set()


def test_long_repeated_lines_are_not_headers():
    """Repeated prose is content. Only short lines qualify as page furniture."""
    para = "This long sentence repeats across pages but is genuine body prose that the " \
           "reader is meant to see, so it must never be stripped as page furniture ok."
    assert len(para) > pp._HEADER_MAX_CHARS
    assert pp._find_running_headers([f"{para}\nbody {i}" for i in range(6)]) == set()


def test_header_counted_once_per_page():
    """A phrase repeated many times on ONE page is not a running header."""
    pages = ["Note\nNote\nNote\nNote\nbody a", "body b", "body c", "body d"]
    assert "Note" not in pp._find_running_headers(pages)


def test_clean_page_removes_both_kinds_of_furniture():
    page = _numbered_page(100, ["Venue Header"] + [f"Real content line {i}." for i in range(14)])
    cleaned = pp._clean_page(page, {"Venue Header"})
    assert "Venue Header" not in cleaned
    assert "100" not in cleaned
    assert "Real content line 9." in cleaned
