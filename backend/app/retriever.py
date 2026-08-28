"""
Retriever Module, Build FAISS index and retrieve relevant chunks.

Embeddings arriving here are L2-normalized (see embedder.py), so an inner-product
index computes true cosine similarity, the metric all five offered embedding
models are trained for. Scores are in [-1, 1] and are NOT probabilities.
"""

import numpy as np
import faiss
from typing import List, Dict
from app.embedder import embed_query


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS inner-product (cosine, given unit-norm inputs) index.

    Args:
        embeddings: numpy array of shape (n, dim), rows L2-normalized

    Returns:
        FAISS index
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    embedding_model: str,
    top_k: int = 3,
) -> List[Dict]:
    """
    Retrieve the most relevant chunks for a given query.

    Args:
        query: The user's question.
        index: FAISS index of chunk embeddings.
        chunks: Original text chunks.
        embedding_model: Model used for embedding.
        top_k: Number of chunks to retrieve.

    Returns:
        List of dicts with chunk text, index, and cosine similarity
        (in [-1, 1]; higher is more similar; not a probability).
    """
    query_embedding = embed_query(query, embedding_model)
    similarities, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = []
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
        if 0 <= idx < len(chunks):
            results.append({
                "rank": i + 1,
                "index": int(idx),
                "text": chunks[idx],
                "similarity": float(sim),
            })

    return results
