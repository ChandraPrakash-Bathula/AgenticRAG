"""Embedder behavior that needs no model download: prefixes and fail-loud reduction."""
import numpy as np
import pytest

from app import embedder


def test_model_query_prefixes_applied():
    class Spy:
        def __init__(self):
            self.seen = []

        def encode(self, texts, **kw):
            self.seen.extend(texts)
            return np.ones((len(texts), 4), dtype=np.float32)

    cases = [
        ("BAAI/bge-base-en-v1.5", "", "Represent this sentence for searching relevant passages: "),
        ("intfloat/e5-large-v2", "passage: ", "query: "),
        ("nomic-ai/nomic-embed-text-v1.5", "search_document: ", "search_query: "),
        ("sentence-transformers/all-mpnet-base-v2", "", ""),
    ]
    for model_name, want_doc, want_q in cases:
        spy = Spy()
        embedder._model_cache[model_name] = spy
        try:
            embedder.embed_chunks(["doc text"], model_name)
            embedder.embed_query("question", model_name)
        finally:
            del embedder._model_cache[model_name]
        assert spy.seen == [f"{want_doc}doc text", f"{want_q}question"], model_name


def test_reduction_fails_loudly_on_nan():
    bad = np.full((10, 8), np.nan, dtype=np.float32)
    with pytest.raises(Exception):
        embedder.reduce_dimensions_2d(bad)


def test_no_random_fallback_in_source():
    src = open(embedder.__file__).read()
    assert "np.random.normal" not in src and "np.random.seed" not in src
