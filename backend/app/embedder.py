"""
Embedding Module, Generate embeddings using sentence-transformers
and reduce dimensions for visualization.

Fixes SSL_CERT_FILE env var issue on macOS before any imports that use httpx.
"""

import os
import sys

# ── Fix SSL cert path before any HuggingFace / httpx imports ─────────────────
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.exists(_ssl_cert):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from typing import List

# Cache loaded models
_model_cache = {}

EMBEDDING_MODELS = [
    {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "explainer": "A 110M-parameter transformer fine-tuned on over a billion sentence pairs. The long-standing 'best all-round' sentence-transformer: strong semantic similarity across many domains, 768-dimensional vectors, cosine-trained, no special prefixes needed. A safe default when you don't know your domain yet.",
        "download_size": "~420 MB",
        "name": "MPNet-Base-v2",
        "description": "Top-tier general-purpose model. 768 dimensions, best overall quality on SBERT benchmarks.",
        "dim": 768,
        "icon": "🏆",
    },
    {
        "id": "BAAI/bge-base-en-v1.5",
        "explainer": "Beijing Academy of AI's retrieval-specialized model, trained specifically for query-to-passage search and among the top open models on the MTEB retrieval benchmark. Its documented query instruction ('Represent this sentence for searching relevant passages:') is required for best results, this app adds it automatically.",
        "download_size": "~440 MB",
        "name": "BGE-Base-EN v1.5",
        "description": "BAAI's high-performing open-source model. Strong retrieval scores, 768 dimensions.",
        "dim": 768,
        "icon": "🔬",
    },
    {
        "id": "intfloat/e5-large-v2",
        "explainer": "Microsoft's contrastively-trained retriever, the strongest retrieval quality of the five, and the largest (1024-dimensional, ~1.3 GB download). Requires 'query:' / 'passage:' prefixes on inputs, which this app applies for you. Choose it to see whether quality gains justify the size and latency.",
        "download_size": "~1.3 GB",
        "name": "E5-Large-v2",
        "description": "Microsoft's E5 model. Best open-source retrieval quality, 1024 dimensions.",
        "dim": 1024,
        "icon": "⚡",
    },
    {
        "id": "nomic-ai/nomic-embed-text-v1.5",
        "explainer": "A fully open model, weights, training data, and code are all published, which is rare. Uses task prefixes (search_document / search_query, applied automatically) and supports long inputs. A good example of a modern, auditable open embedding model.",
        "download_size": "~550 MB",
        "name": "Nomic-Embed-Text v1.5",
        "description": "Nomic AI's fully open-source embedding. Excellent cost-performance, 768 dimensions.",
        "dim": 768,
        "icon": "🌐",
    },
    {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "explainer": "A 22M-parameter distilled model: several times faster than the others with compact 384-dimensional vectors and surprisingly solid quality. The default for prototyping and for this app's exercises, compare it against E5-Large to feel the speed/quality trade-off.",
        "download_size": "~90 MB",
        "name": "MiniLM-L6-v2",
        "description": "Lightweight & fast. 384 dimensions, great for quick prototyping.",
        "dim": 384,
        "icon": "🚀",
    },
]


def is_model_cached(model_name: str) -> bool:
    """True if the model's weights are already on disk (loaded, or present in the
    HuggingFace hub cache), i.e. selecting it will NOT trigger a large download."""
    if model_name in _model_cache:
        return True
    from pathlib import Path
    hf_home = os.environ.get("HF_HOME")
    cache_root = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    snapshot_dir = cache_root / "hub" / ("models--" + model_name.replace("/", "--"))
    return snapshot_dir.is_dir()


def _get_model(model_name: str) -> SentenceTransformer:
    """Load and cache a sentence-transformer model."""
    if model_name not in _model_cache:
        kwargs = {}
        # nomic-embed-text requires trust_remote_code
        if "nomic" in model_name.lower():
            kwargs["trust_remote_code"] = True
        _model_cache[model_name] = SentenceTransformer(model_name, **kwargs)
    return _model_cache[model_name]


# Per-model usage requirements from each model card. All five models are trained for
# COSINE similarity, so every embedding is L2-normalized (paired with an inner-product
# FAISS index in retriever.py). E5, BGE, and Nomic additionally require task prefixes,
# skipping them silently degrades retrieval and invalidates cross-model comparisons.
_DOC_PREFIXES = {
    "e5": "passage: ",
    "nomic": "search_document: ",
}
_QUERY_PREFIXES = {
    "e5": "query: ",
    "bge": "Represent this sentence for searching relevant passages: ",
    "nomic": "search_query: ",
}


def _prefix_for(model_name: str, prefixes: dict) -> str:
    lowered = model_name.lower()
    for key, prefix in prefixes.items():
        if key in lowered:
            return prefix
    return ""


def embed_chunks(chunks: List[str], model_name: str) -> np.ndarray:
    """
    Generate L2-normalized embeddings for text chunks.

    Args:
        chunks: List of text chunks.
        model_name: HuggingFace model identifier.

    Returns:
        numpy array of shape (num_chunks, embedding_dim), unit-norm rows
    """
    model = _get_model(model_name)
    prefix = _prefix_for(model_name, _DOC_PREFIXES)
    prefixed = [f"{prefix}{c}" for c in chunks] if prefix else chunks

    embeddings = model.encode(
        prefixed, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
    )
    return embeddings.astype(np.float32)


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Embed a single query string (L2-normalized, with the model's query prefix)."""
    model = _get_model(model_name)
    prefix = _prefix_for(model_name, _QUERY_PREFIXES)
    if prefix:
        query = f"{prefix}{query}"

    embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def _reduce_dimensions(embeddings: np.ndarray, n_components: int) -> np.ndarray:
    """
    Robustly reduce embeddings to lower dimensions (n_components).
    Handles small sample sizes, UMAP failure modes, t-SNE, and PCA fallbacks.
    """
    n_samples = embeddings.shape[0]

    # Handle single sample case (PCA division-by-zero warnings)
    if n_samples == 1:
        return np.zeros((1, n_components), dtype=np.float32)

    # Handle tiny sample sizes with PCA + zero-padding (a real projection, just low-rank)
    if n_samples < n_components + 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(n_components, n_samples))
        reduced = pca.fit_transform(embeddings)
        if reduced.shape[1] < n_components:
            padding = np.zeros((n_samples, n_components - reduced.shape[1]), dtype=np.float32)
            reduced = np.hstack([reduced, padding])
        return reduced.astype(np.float32)

    # Try UMAP with spectral initialization
    try:
        from umap import UMAP
        reducer = UMAP(
            n_components=n_components,
            random_state=42,
            n_neighbors=min(15, n_samples - 1),
            init="spectral"
        )
        return reducer.fit_transform(embeddings)
    except Exception:
        # If spectral initialization fails (e.g., scipy.linalg.eigh error), try random initialization
        try:
            from umap import UMAP
            reducer = UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=min(15, n_samples - 1),
                init="random"
            )
            return reducer.fit_transform(embeddings)
        except Exception:
            pass

    # Try t-SNE
    try:
        perplexity = min(30, max(2, n_samples - 1))
        if perplexity >= n_samples:
            perplexity = max(1, n_samples - 1)
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        return tsne.fit_transform(embeddings)
    except Exception:
        pass

    # Try PCA
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(embeddings)
    except Exception as e:
        # NO fake fallback. Rendering random points as "the embedding space" teaches
        # learners to read structure into noise, fail loudly instead.
        raise RuntimeError(
            f"Dimensionality reduction failed (UMAP, t-SNE, and PCA all errored): {e}"
        ) from e


def reduce_dimensions_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D using robust multi-stage reduction."""
    return _reduce_dimensions(embeddings, 2)


def reduce_dimensions_3d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 3D using robust multi-stage reduction."""
    return _reduce_dimensions(embeddings, 3)

