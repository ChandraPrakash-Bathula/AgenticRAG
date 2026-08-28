"""Chunking strategies: parameter effects and invariants."""
from app.chunker import chunk_text

LONG_SENTENCE = "This sentence is deliberately written to be about seventy chars long."
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 60


def test_min_sentences_has_effect():
    text = " ".join([LONG_SENTENCE] * 30)
    c1 = chunk_text(text, "sentence", chunk_size=100, min_sentences=1)
    c3 = chunk_text(text, "sentence", chunk_size=100, min_sentences=3)
    assert c1 != c3
    assert all(ch.count(".") == 1 for ch in c1)
    for ch in c3[:-1]:
        assert ch.count(".") >= 3


def test_recursive_overlap_respects_size_and_word_boundaries():
    chunks = chunk_text(WORDS, "recursive", chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 2
    vocab = set(WORDS.split())
    for ch in chunks:
        assert len(ch) <= 200
        assert ch.split(" ", 1)[0] in vocab, f"mid-word slice: {ch[:30]!r}"
    no_overlap = chunk_text(WORDS, "recursive", chunk_size=200, chunk_overlap=0)
    assert any(len(a) > len(b) for a, b in zip(chunks[1:], no_overlap[1:]))


def test_fixed_chunking_covers_text():
    chunks = chunk_text(WORDS, "fixed", chunk_size=300, chunk_overlap=30)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks).count("alpha") >= WORDS.count("alpha")  # overlap repeats, never drops


def test_semantic_chunking_merges_short_paragraphs():
    text = "Tiny.\n\n" + ("A regular paragraph with a decent amount of text in it. " * 3) + "\n\nAlso tiny."
    chunks = chunk_text(text, "semantic", chunk_size=800, min_paragraph_length=50)
    assert len(chunks) >= 1
    assert "Tiny." in " ".join(chunks)


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", "quantum")
