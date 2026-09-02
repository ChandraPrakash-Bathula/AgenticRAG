"""
RAG Pipeline Backend, FastAPI Application
Handles PDF processing, chunking, embedding, retrieval, and LLM inference.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file before any other imports
import json
import tempfile
from pathlib import Path
import numpy as np
from collections import OrderedDict
from threading import Lock
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_text, CHUNKING_STRATEGIES
from app.embedder import (
    embed_chunks,
    EMBEDDING_MODELS,
    is_model_cached,
    reduce_dimensions_2d,
    reduce_dimensions_3d,
)
from app.retriever import build_index, retrieve_chunks
from app.llm_handler import (
    query_llm,
    get_available_llms,
    check_ollama_status,
    is_model_available,
    get_fast_model_id,
    get_fast_model_hint,
    LLMRateLimitError,
    LLMUnavailableError,
)
from app.agentic_rag import run_agentic_pipeline
from app.example_data import get_example_bundle

app = FastAPI(title="RAG Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session state, scoped per client ───────────────────────────────
# Each browser tab sends a stable X-Session-Id header (see frontend/src/config.js),
# and gets its own pipeline state, concurrent users no longer clobber each other.
# State is still in-memory and single-process: it does not survive restarts and
# does not work behind a multi-worker/multi-replica deployment (documented in README).
MAX_SESSIONS = 50  # LRU cap so abandoned sessions don't leak memory

_sessions: "OrderedDict[str, dict]" = OrderedDict()
_sessions_lock = Lock()
# IDs evicted from the LRU (bounded). Lets us tell a returning evicted user "your
# session expired" instead of the misleading "No PDF uploaded yet".
_evicted_ids: set = set()
MAX_EVICTED_IDS = 1000


def _blank_session() -> dict:
    return {
        "pdf_text": None,
        "pdf_filename": None,
        "chunks": None,
        "chunk_strategy": None,
        "embeddings": None,
        "embedding_model": None,
        "faiss_index": None,
        "selected_llm": None,
    }


def get_session(x_session_id: Optional[str] = Header(default=None)) -> dict:
    """FastAPI dependency: return (creating if needed) the caller's session state."""
    key = x_session_id or "default"
    with _sessions_lock:
        if key not in _sessions:
            fresh = _blank_session()
            if key in _evicted_ids:
                _evicted_ids.discard(key)
                fresh["_was_evicted"] = True
            _sessions[key] = fresh
        _sessions.move_to_end(key)
        while len(_sessions) > MAX_SESSIONS:
            old_key, _ = _sessions.popitem(last=False)
            _evicted_ids.add(old_key)
        while len(_evicted_ids) > MAX_EVICTED_IDS:
            _evicted_ids.pop()
        return _sessions[key]


def _require_state(session: dict, field: str, missing_msg: str) -> None:
    """Guard for endpoints that need earlier pipeline state. Distinguishes 'you
    skipped a step' (400) from 'your session was evicted' (410, actionable)."""
    if session[field] is not None:
        return
    if session.get("_was_evicted"):
        raise HTTPException(
            status_code=410,
            detail=(
                "Your session expired on the server (evicted after inactivity or a "
                "restart), please rerun the pipeline from Step 1."
            ),
        )
    raise HTTPException(status_code=400, detail=missing_msg)


# ── Pydantic models ──────────────────────────────────────────────────────────
class ChunkRequest(BaseModel):
    strategy: str
    chunk_size: Optional[int] = 500
    chunk_overlap: Optional[int] = 50
    min_sentences: Optional[int] = 1
    min_paragraph_length: Optional[int] = 50


class EmbedRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str


class LLMSelectRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


class AgenticQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)
    max_loops: int = Field(default=3, ge=1, le=5)
    relevance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    demo_mode: Optional[bool] = False


# ── Step 1: PDF Upload ───────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


ALLOWED_UPLOAD_EXTENSIONS = (".pdf", ".txt", ".md")


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), session: dict = Depends(get_session)):
    """Upload a document: PDF (text extracted via PyMuPDF), or plain .txt/.md."""
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(ALLOWED_UPLOAD_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, TXT, or Markdown (.md) files are accepted.",
        )

    size_error = HTTPException(
        status_code=413,
        detail=f"File too large, maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise size_error
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise size_error

    tmp_path = None
    try:
        if filename_lower.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            text, page_texts = extract_text_from_pdf(tmp_path)
        else:
            # Plain text / Markdown, no extraction needed, treat as a single "page".
            text = contents.decode("utf-8", errors="replace")
            page_texts = [text]

        session["pdf_text"] = text
        session["pdf_filename"] = file.filename
        # Reset downstream
        session["chunks"] = None
        session["embeddings"] = None
        session["faiss_index"] = None

        return {
            "filename": file.filename,
            "total_pages": len(page_texts),
            "total_characters": len(text),
            "preview": text[:2000],
            "page_previews": [
                {"page": i + 1, "preview": p[:300]} for i, p in enumerate(page_texts[:5])
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.post("/api/load-example")
def load_example(session: dict = Depends(get_session)):
    """Load the bundled worked example (an excerpt of the open textbook 'Human
    Nutrition: 2020 Edition') straight into this session's pipeline state, real
    chunks, real embeddings, real FAISS index, computed once per server process
    and reused after that. Lets a first-time visitor see the whole pipeline
    (and the plain-LLM vs naive-RAG vs agentic-RAG contrast) without sourcing
    their own document or hand-picking good questions first."""
    try:
        bundle = get_example_bundle()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load worked example: {str(e)}")

    session["pdf_text"] = bundle["text"]
    session["pdf_filename"] = bundle["upload"]["filename"]
    session["chunks"] = bundle["chunks"]
    session["chunk_strategy"] = bundle["chunk"]["strategy"]
    session["embeddings"] = bundle["embeddings"]
    session["embedding_model"] = bundle["embedding_model"]
    session["faiss_index"] = bundle["faiss_index"]

    return {
        "upload": bundle["upload"],
        "chunk": bundle["chunk"],
        "embed": bundle["embed"],
        "questions": bundle["questions"],
    }


# ── Step 2: Chunking ─────────────────────────────────────────────────────────
@app.get("/api/chunking-strategies")
def get_chunking_strategies():
    """Return available chunking strategies."""
    return {"strategies": CHUNKING_STRATEGIES}


@app.post("/api/chunk")
def chunk_document(req: ChunkRequest, session: dict = Depends(get_session)):
    """Chunk the uploaded PDF text."""
    _require_state(session, "pdf_text", "No document uploaded yet.")

    # Layer 1: Validate the document has enough content to chunk.
    text = session["pdf_text"]
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="The uploaded document contains no extractable text.",
        )
    if len(text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="The uploaded document is too short to chunk (minimum 10 characters).",
        )

    chunks = chunk_text(
        text,
        req.strategy,
        req.chunk_size,
        req.chunk_overlap,
        req.min_sentences,
        req.min_paragraph_length,
    )

    # Layer 2: Guard against chunkers that return an empty list.
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Chunking produced no chunks with strategy '{req.strategy}'. "
                "Try a different strategy or adjust the chunking parameters."
            ),
        )

    session["chunks"] = chunks
    session["chunk_strategy"] = req.strategy
    # Reset downstream
    session["embeddings"] = None
    session["faiss_index"] = None

    return {
        "strategy": req.strategy,
        "total_chunks": len(chunks),
        "avg_chunk_length": round(np.mean([len(c) for c in chunks]), 1),
        "min_chunk_length": min(len(c) for c in chunks),
        "max_chunk_length": max(len(c) for c in chunks),
        "chunks_preview": [
            {"index": i, "text": c[:200], "length": len(c)}
            for i, c in enumerate(chunks[:20])
        ],
    }



# ── Step 3: Embedding ────────────────────────────────────────────────────────
@app.get("/api/embedding-models")
def get_embedding_models():
    """Return available embedding models, flagging which are already downloaded,
    an uncached model means a large blocking download on first use."""
    return {
        "models": [
            {**m, "cached": is_model_cached(m["id"])} for m in EMBEDDING_MODELS
        ]
    }


@app.post("/api/embed")
def embed_document(req: EmbedRequest, session: dict = Depends(get_session)):
    """Embed chunks using the selected model."""
    _require_state(session, "chunks", "No chunks available. Run chunking first.")

    try:
        embeddings = embed_chunks(session["chunks"], req.model_name)
        session["embeddings"] = embeddings
        session["embedding_model"] = req.model_name

        # Build FAISS index
        session["faiss_index"] = build_index(embeddings)

        # Dimensionality reduction for visualization. If it fails we return an explicit
        # error state, never fabricated coordinates. Embeddings + index stay usable.
        coords_2d = coords_3d = None
        viz_error = None
        try:
            coords_2d = reduce_dimensions_2d(embeddings)
            coords_3d = reduce_dimensions_3d(embeddings)
        except Exception as viz_exc:
            viz_error = f"Embedding visualization unavailable, {viz_exc}"

        return {
            "model": req.model_name,
            "embedding_dim": embeddings.shape[1],
            "num_embeddings": embeddings.shape[0],
            "visualization_2d": coords_2d.tolist() if coords_2d is not None else None,
            "visualization_3d": coords_3d.tolist() if coords_3d is not None else None,
            "visualization_error": viz_error,
            "chunk_labels": [
                session["chunks"][i][:60] for i in range(len(session["chunks"]))
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


# ── Step 4: LLM Selection ────────────────────────────────────────────────────
def describe_document(session: dict, samples: int = 4, sample_chars: int = 320) -> str:
    """A short description of the uploaded document, for the agentic router.

    The router decides retrieve-or-not from the question alone unless it is told what
    the document is about, which made it skip retrieval for on-topic questions that
    simply never used the word "document".

    Samples are spread EVENLY across the text rather than taken from the opening. A
    600-char head of a 1,200-page textbook describes chapter one and nothing else, and
    a router shown only that will reject on-topic questions about every other chapter
    as "unrelated", which is worse than having no description at all."""
    text = " ".join((session.get("pdf_text") or "").split())
    name = session.get("pdf_filename") or "uploaded document"
    if not text:
        return ""

    if len(text) <= samples * sample_chars:
        excerpts = [text]
    else:
        # Span the FULL range: offsets run from 0 to (len - sample_chars), so the last
        # sample lands at the document's tail. Stepping by len//samples instead would
        # stop ~one step short and never sample the final section.
        span = len(text) - sample_chars
        excerpts = [
            text[round(i * span / (samples - 1)):][:sample_chars].strip()
            for i in range(samples)
        ]

    body = "\n".join(f"  […] {e} […]" for e in excerpts)
    return (
        f"Filename: {name}\n"
        f"Length: ~{len(text):,} characters\n"
        f"Excerpts sampled from across the document (a small fraction of the whole):\n{body}"
    )


@app.get("/api/llm-models")
def get_llm_models():
    """Return available LLM models."""
    # Preflight for agentic mode: is the fast grader model actually available?
    # (In Ollama mode students often haven't pulled it, the pipeline falls back to
    # their selected model, but the UI should tell them up front, not mid-query.)
    fast_model = get_fast_model_id()
    grader_available = is_model_available(fast_model)
    return {
        "models": get_available_llms(),
        "ollama_status": check_ollama_status(),
        "agentic_grader": {
            "model": fast_model,
            "available": grader_available,
            "note": None if grader_available else (
                "The agentic pipeline's fast grader model isn't installed, your selected "
                f"model will be used for grading instead (slower). Fix: {get_fast_model_hint()}"
            ),
        },
    }


@app.post("/api/select-llm")
def select_llm(req: LLMSelectRequest, session: dict = Depends(get_session)):
    """Select an LLM for inference."""
    session["selected_llm"] = req.model_name
    return {"selected_model": req.model_name}


# ── Step 5: Query ────────────────────────────────────────────────────────────
@app.post("/api/query")
def query(req: QueryRequest, session: dict = Depends(get_session)):
    """Run a query: retrieve relevant chunks and generate answers."""
    _require_state(session, "faiss_index", "Build embeddings first.")
    _require_state(session, "selected_llm", "Select an LLM first.")

    try:
        # Retrieve relevant chunks
        retrieved = retrieve_chunks(
            req.question,
            session["faiss_index"],
            session["chunks"],
            session["embedding_model"],
            top_k=req.top_k,
        )

        context = "\n\n---\n\n".join([r["text"] for r in retrieved])

        # Plain LLM answer (no context), a baseline shown alongside, not part of the RAG cost.
        # It gets its own stats sink solely to detect token-limit truncation.
        plain_stats = []
        plain_answer = query_llm(
            session["selected_llm"],
            req.question,
            context=None,
            stats_sink=plain_stats,
            call_label="plain",
        )

        # RAG answer (with context), the call whose cost is comparable to the agentic pipeline.
        stats = []
        rag_answer = query_llm(
            session["selected_llm"],
            req.question,
            context=context,
            stats_sink=stats,
            call_label="generate",
        )
        total_tokens = [s["totalTokens"] for s in stats if s.get("totalTokens") is not None]

        return {
            "question": req.question,
            "plain_answer": plain_answer,
            "rag_answer": rag_answer,
            "plain_truncated": any(s.get("truncated") for s in plain_stats),
            "rag_truncated": any(s.get("truncated") for s in stats),
            "retrieved_chunks": retrieved,
            "source_file": session["pdf_filename"],
            "stats": {
                "calls": len(stats),
                "latencyMs": round(sum(s["elapsedMs"] for s in stats), 1),
                "totalTokens": sum(total_tokens) if total_tokens else None,
            },
        }
    except LLMRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except LLMUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/query-agentic")
def query_agentic(req: AgenticQueryRequest, session: dict = Depends(get_session)):
    """Run the self-correcting Agentic RAG pipeline (route → retrieve → grade → generate → grade → loop)."""
    _require_state(session, "faiss_index", "Build embeddings first.")
    _require_state(session, "selected_llm", "Select an LLM first.")

    try:
        result = run_agentic_pipeline(
            session["selected_llm"],
            req.question,
            session["faiss_index"],
            session["chunks"],
            session["embedding_model"],
            top_k=req.top_k,
            max_loops=req.max_loops,
            relevance_threshold=req.relevance_threshold,
            document_context=describe_document(session),
            demo_mode=req.demo_mode,
        )
        return {
            "question": req.question,
            "source_file": session["pdf_filename"],
            **result,
        }
    except LLMRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except LLMUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agentic query failed: {str(e)}")


@app.get("/api/health")
def health():
    return {"status": "ok", "ollama": check_ollama_status()}


# ── Single-port static serving (Hugging Face Spaces, Docker, any one-port host) ──
# A Space exposes exactly ONE port, so the built React app has to come from this
# process rather than a separate Vite server. Mounted LAST, after every /api route is
# registered, so it can never shadow the API. If the bundle is absent (normal local
# development, where Vite serves the frontend on 5173) this block is skipped entirely
# and nothing about the dev workflow changes.
# Repo layout puts the bundle at <repo>/frontend/dist. The container copies backend/ to
# /app and the bundle to /frontend/dist, which this same expression happens to resolve
# to, but relying on that coincidence is fragile, so FRONTEND_DIST can state it outright.
_FRONTEND_DIST = Path(
    os.environ.get("FRONTEND_DIST")
    or Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
)

if _FRONTEND_DIST.is_dir():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        """Serve the SPA, falling back to index.html for client-side routes.

        Unknown /api paths must still 404 as JSON rather than silently returning the
        HTML shell, otherwise a typo in a fetch looks like a parse error to the caller.
        """
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        candidate = (_FRONTEND_DIST / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_FRONTEND_DIST):
            return FileResponse(candidate)
        return FileResponse(_FRONTEND_DIST / "index.html")
