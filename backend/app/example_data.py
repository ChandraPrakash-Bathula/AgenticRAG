"""
Worked Example, "Human Nutrition: 2020 Edition" (Univ. of Hawaiʻi at Mānoa, open
textbook) excerpt covering Chapters 3-6 (Water & Electrolytes, Carbohydrates,
Lipids, Protein).

Why this exists: the live pipeline (upload -> chunk -> embed -> query) is the
whole point of the app, but asking a first-time visitor to source their own PDF
and hand-pick good questions is a lot of friction before they see the payoff.
This module ships a small pre-extracted excerpt of a real textbook and a
curated question set chosen specifically because the facts in them are exact,
textbook-specific numbers (e.g. "3,000 kilocalories of glycogen in muscle") that
a plain LLM tends to guess or generalize on, naive RAG retrieves correctly, and
agentic RAG additionally verifies before answering.

Everything downstream of loading is still the REAL pipeline, real chunking,
real sentence-transformer embeddings, a real FAISS index, and real LLM calls at
query time. Nothing about the answers themselves is scripted or precomputed.
"""

import json
from pathlib import Path
from threading import Lock
from typing import Dict

import numpy as np

from app.chunker import chunk_text
from app.embedder import embed_chunks
from app.retriever import build_index

_DATA_DIR = Path(__file__).parent / "data"
_TEXT_PATH = _DATA_DIR / "nutrition_excerpt.txt"
_META_PATH = _DATA_DIR / "nutrition_excerpt_meta.json"

DISPLAY_FILENAME = "Human_Nutrition_excerpt.pdf (worked example)"

# Kept identical to the app's own defaults (recursive / 500 / 50) and its
# pre-baked embedding model (see backend/Dockerfile) so the first "Load Example"
# request never has to download anything, only compute.
CHUNK_STRATEGY = "recursive"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Each question targets a specific, textbook-exact figure, but "textbook-exact"
# is not the same as "obscure enough that a plain LLM can't already recite it".
# Public-health numbers repeated across CDC/NIH/USDA sources (e.g. the 2,300mg
# sodium UL) are usually already baked into a modern LLM's training data, so a
# plain-LLM call gets them right too and the intended gap disappears. These five
# were chosen (and live-verified against this app's own pipeline, not guessed)
# specifically because a plain LLM answers them wrong or refuses outright, while
# retrieval nails the exact figure from the text:
#   - glycogen split: plain LLM guesses "350-400g total" (wrong); text says 4,000
#     kcal, 3,000 in muscle / 1,000 in liver.
#   - cholesterol-in-brain: plain LLM outright declines to answer without the
#     book; text says ~25%.
#   - lecithin composition: plain LLM guesses "~30%" for brain matter (text says
#     28%) and admits it can't find a liver figure (text says 66%).
#   - soybean macros: plain LLM states 0.5g saturated fat (text says 2.2g), a
#     real, substantive factual disagreement, not just missing detail.
#   - sports-drink sodium: plain LLM guesses a "110-170mg" range (text gives an
#     exact, different figure: 125mg).
EXAMPLE_QUESTIONS = [
    {
        "question": "According to this book, how many kilocalories of glycogen does the body store, and how is that split between muscle and liver?",
        "chapter": "Chapter 4, Carbohydrates",
        "expectedFact": "~4,000 kcal total: about 3,000 kcal in muscle tissue and 1,000 kcal in the liver.",
    },
    {
        "question": "According to this book, what percentage of the body's cholesterol is located in brain tissue?",
        "chapter": "Chapter 5, Lipids",
        "expectedFact": "Approximately 25 percent.",
    },
    {
        "question": "According to this book, what percentage of brain matter is composed of lecithin, and what percentage of the fat in the liver is lecithin?",
        "chapter": "Chapter 5, Lipids",
        "expectedFact": "28 percent of brain matter; 66 percent of liver fat.",
    },
    {
        "question": "According to this book, how many grams of protein, saturated fat, and cholesterol does a cup of boiled soybeans contain?",
        "chapter": "Chapter 6, Protein",
        "expectedFact": "29 grams of protein, 2.2 grams of saturated fat, and no cholesterol.",
    },
    {
        "question": "According to this book, how many milligrams of sodium should a sports drink contain per 8 ounces, per the American College of Sports Medicine?",
        "chapter": "Chapter 3, Water & Electrolytes",
        "expectedFact": "125 milligrams per 8 ounces.",
    },
    # ── Out-of-corpus questions ──────────────────────────────────────────────
    # The five above are answerable from a single chunk, so retrieval succeeds on the
    # first try and the self-correction loop never fires: they demonstrate the
    # plain-LLM vs RAG gap, not the naive vs agentic one. These two ask about nutrients
    # that appear NOWHERE in the Ch 3-6 excerpt, which is where the loop runs for real.
    #
    # Verified absent by substring count against app/data/nutrition_excerpt.txt. Do not
    # swap in calcium, iron, zinc, magnesium or vitamins A/C/D/E: all of those ARE
    # mentioned in passing, so retrieval returns plausible neighbouring tables and the
    # model misattributes a nearby figure (sodium's 1,500 mg/day was being reported as
    # a calcium recommendation, and the answer graded as "supported"). Any replacement
    # must be grepped against the corpus first.
    {
        "question": "According to this book, what is the recommended daily intake of riboflavin?",
        "chapter": "Not in this excerpt (vitamins are Ch. 7)",
        "expectedFact": "Nothing. Riboflavin appears nowhere in Chapters 3-6, so the honest answer is that this excerpt does not say. Watch whether each mode admits that or invents a number.",
        "outOfCorpus": True,
    },
    {
        "question": "According to this book, how much vitamin K do adults need per day?",
        "chapter": "Not in this excerpt (vitamins are Ch. 7)",
        "expectedFact": "Nothing. Vitamin K appears nowhere in Chapters 3-6. Compare how each mode signals that it could not find an answer, and at what cost.",
        "outOfCorpus": True,
    },
]

_lock = Lock()
_cache: Dict = {}


def _load_source() -> tuple[str, dict]:
    text = _TEXT_PATH.read_text(encoding="utf-8")
    meta = json.loads(_META_PATH.read_text(encoding="utf-8"))
    return text, meta


def get_example_bundle() -> Dict:
    """Return the worked-example pipeline state, computing (and caching, for the
    life of this process) chunks/embeddings/index on first call. Identical for
    every session, this is shared reference content, not per-user state."""
    with _lock:
        if _cache:
            return _cache

        text, meta = _load_source()

        chunks = chunk_text(text, CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP)
        embeddings = embed_chunks(chunks, EMBEDDING_MODEL)
        faiss_index = build_index(embeddings)

        upload = {
            "filename": DISPLAY_FILENAME,
            "total_pages": meta["total_pages"],
            "total_characters": meta["total_characters"],
            "preview": meta["preview"],
            "page_previews": meta["page_previews"],
            "note": (
                f"Worked example: {meta['display_name']} "
                f"(original pages {meta['source_page_range'][0]}"
                f"–{meta['source_page_range'][1]})."
            ),
        }

        chunk_payload = {
            "strategy": CHUNK_STRATEGY,
            "total_chunks": len(chunks),
            "avg_chunk_length": round(float(np.mean([len(c) for c in chunks])), 1),
            "min_chunk_length": min(len(c) for c in chunks),
            "max_chunk_length": max(len(c) for c in chunks),
            "chunks_preview": [
                {"index": i, "text": c[:200], "length": len(c)}
                for i, c in enumerate(chunks[:20])
            ],
        }

        embed_payload = {
            "model": EMBEDDING_MODEL,
            "embedding_dim": embeddings.shape[1],
            "num_embeddings": embeddings.shape[0],
            # Visualization is skipped here on purpose: UMAP/t-SNE over ~700 points
            # adds real compute to every cold start for a view most visitors of
            # the worked example won't linger on. The step still shows real
            # stats; only the scatter plot is omitted, with an honest note why.
            "visualization_2d": None,
            "visualization_3d": None,
            "visualization_error": (
                "Skipped for the worked example to keep first load fast, the "
                "embeddings and retrieval index above are real and fully queryable."
            ),
            "chunk_labels": [c[:60] for c in chunks],
        }

        _cache.update({
            "text": text,
            "chunks": chunks,
            "embeddings": embeddings,
            "embedding_model": EMBEDDING_MODEL,
            "faiss_index": faiss_index,
            "upload": upload,
            "chunk": chunk_payload,
            "embed": embed_payload,
            "questions": EXAMPLE_QUESTIONS,
        })
        return _cache
