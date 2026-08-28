"""Retrieval correctness with a real embedding model. Skipped automatically when the
model isn't in the local HF cache (e.g. bare CI) unless RUN_MODEL_TESTS=1 forces the
download."""
import os

import numpy as np
import pytest

from app import embedder, retriever

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

pytestmark = pytest.mark.skipif(
    not embedder.is_model_cached(MODEL) and os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="MiniLM not in HF cache; set RUN_MODEL_TESTS=1 to download and run",
)

CHUNKS = [
    "The mitochondria is the powerhouse of the cell, producing ATP through respiration.",
    "Stock markets fell sharply on Tuesday amid concerns about interest rate hikes.",
    "Cellular respiration converts glucose and oxygen into usable chemical energy.",
    "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
    "Photosynthesis in plant cells captures light energy to synthesize sugars.",
]
QUERY = "How do cells generate energy?"


def test_embeddings_are_unit_norm():
    emb = embedder.embed_chunks(CHUNKS, MODEL)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    q = embedder.embed_query(QUERY, MODEL)
    assert np.allclose(np.linalg.norm(q), 1.0, atol=1e-5)


def test_ranking_matches_exact_cosine():
    emb = embedder.embed_chunks(CHUNKS, MODEL)
    q = embedder.embed_query(QUERY, MODEL)
    index = retriever.build_index(emb)
    results = retriever.retrieve_chunks(QUERY, index, CHUNKS, MODEL, top_k=5)

    ref = (emb @ q[0]) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(q[0]))
    assert [r["index"] for r in results] == list(np.argsort(-ref))
    for r in results:
        assert abs(r["similarity"] - ref[r["index"]]) < 1e-5
        assert -1.0 <= r["similarity"] <= 1.0
        assert "relevance_score" not in r and "distance" not in r
