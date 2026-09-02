# Self-Correcting-RAG Pipeline Studio

**Self-correcting RAG, interactively, learn the reflective retrieval loop (Self-RAG / CRAG) by first building the naive pipeline it fixes**

![RAG Pipeline](https://img.shields.io/badge/RAG-Pipeline%20Studio-blueviolet?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)

An interactive lesson on **self-corrective retrieval**, the reflect → verify → retry loop from Self-RAG (ICLR 2024) and CRAG (2024) that today's agentic search systems build on. You first assemble a working naive RAG pipeline step by step (upload → chunk → embed → retrieve → generate) as the baseline, then flip one toggle and watch the same question run through the self-correcting loop, every routing decision, evidence grade, support verdict, reformulation, and its token/latency cost rendered live.

The app has three sections: an **Overview** explaining what each pipeline stage teaches and why,
a **Mechanisms** page with side-by-side figures of Self-RAG and CRAG as published versus what this
code actually runs (every simplification named), and the **Playground** itself.

Guided exercises with solutions (three difficulty levels) live in [`exercises/EXERCISES.md`](exercises/EXERCISES.md).

## ✨ Features

### 5-Step Interactive Wizard
1. **📄 Upload PDF**, Drag & drop any PDF, text extracted via PyMuPDF
2. **✂️ Chunking**, 4 strategies with strategy-specific parameters:
   - Fixed-Size (chunk_size + overlap)
   - Recursive Character Splitting (chunk_size + overlap)
   - Sentence-Based (chunk_size + min_sentences)
   - Semantic Paragraph (chunk_size + min_paragraph_length)
3. **🧬 Embedding**, 5 top open-source embedding models:
   - all-mpnet-base-v2, BGE-Base, E5-Large-v2, Nomic-Embed, MiniLM-L6
   - 2D & 3D Plotly visualizations of embedding space
4. **🤖 LLM Selection**, Dual-provider architecture:
   - **Local**: Ollama (Llama 3.2, DeepSeek, Gemma 3, Phi-4, Mistral)
   - **Cloud**: Groq API (Allam 2 7B, GPT-OSS 20B, Qwen 3.6 27B, GPT-OSS 120B)

   > **On "open":** every model above is open-weight and downloadable from HuggingFace.
   > Groq is an inference host, not a model vendor, so the cloud path adds speed, not a
   > proprietary dependency. Retrieval is local either way: PDF text, chunks, embeddings,
   > and the FAISS index are all computed in-process and never leave your machine. Only
   > generation and grading calls hit a model server, and `LLM_PROVIDER` defaults to
   > `ollama`, which keeps those on-device too.
5. **🔍 Query**, Side-by-side comparison of Plain LLM vs RAG answers with source chunks, plus a self-correcting **Agentic RAG** mode with a full decision trace

> **🎬 Demo Mode disclosure:** Agentic mode includes an optional Demo Mode for live teaching:
> it intentionally forces the *first* chunk-grading pass to fail so the self-correction loop
> reliably fires in front of an audience. When active, a persistent banner appears over the
> answer panel and staged trace steps are tagged "🎬 staged". Every step after pass 1 runs for
> real. Demo Mode never affects Naive mode or normal (unchecked) agentic queries.

### Design
- Dark glassmorphism UI with premium aesthetics
- Smooth micro-animations and hover effects
- Fully responsive wizard layout

## 🚀 Running the app

You need **two terminals**: one for the backend API, one for the frontend dev server.
Both must stay running.

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | Check with `python3 --version` |
| **Node.js 18+** | Check with `node --version` |
| Ollama *(optional)* | Only for the fully local path |
| Groq API key *(optional)* | Free at [console.groq.com](https://console.groq.com) |

> **Use `python3`, not `python`.** On macOS `python` often resolves to Apple's Command
> Line Tools build, which will not have your dependencies and fails with
> `No module named uvicorn`.

### Terminal 1: backend

```bash
cd backend

# Create an isolated environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Then pick **one** provider:

**Option A, cloud via Groq (fastest to get running)**

Create `backend/.env`:

```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

The app loads this automatically, so no environment variables are needed on the
command line:

```bash
python3 -m uvicorn app.main:app --port 8000 --reload
```

**Option B, fully local via Ollama (no API key, no network)**

```bash
ollama serve                       # leave running in its own terminal
ollama pull llama3.2               # the generator
ollama pull phi4-mini              # the agentic grader (optional but recommended)

python3 -m uvicorn app.main:app --port 8000 --reload
```

`LLM_PROVIDER` defaults to `ollama`, so no `.env` is required for this path.

Backend is up when [http://localhost:8000/docs](http://localhost:8000/docs) loads.

### Terminal 2: frontend

```bash
cd frontend
npm install
npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)**. The frontend expects the backend
on port 8000.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `No module named uvicorn` | You used `python`. Use `python3`, or activate the venv. |
| `[Errno 48] Address already in use` | A server is already on 8000. `lsof -ti:8000 \| xargs kill` |
| `Build embeddings first` after a restart | Session state is in memory and is lost on restart. Redo the wizard from Step 1. |
| `model_not_found` from Groq | Groq retired that model. Restart the backend to reload the live catalog. |
| Grader warning in Step 4 | Ollama mode without `phi4-mini`. Either `ollama pull phi4-mini` or ignore it; your selected model grades instead, more slowly. |
| SSL / certificate errors during `pip install` | A stale `SSL_CERT_FILE` pointing at a file that no longer exists. Run `unset SSL_CERT_FILE` and retry. |
| `pytest` not found | Dev-only dependency: `pip install pytest` |

> **Do not run with `--workers N`.** Pipeline state is held in memory in a single
> process, so multiple workers break the wizard mid-flow.

## 🏗️ Architecture

```
frontend/ (React + Vite)
└── src/
    ├── App.jsx                  # Overview / playground shell
    ├── App.css                  # Research-paper design system (light, token-driven)
    ├── config.js                # API base URL + per-tab session id
    ├── components/
    │   ├── Home.jsx             # Landing page: topics, the lesson, references
    │   ├── Mechanisms.jsx       # Self-RAG and CRAG: paper vs implementation figures
    │   ├── StepWizard.jsx       # Wizard orchestrator + step gating
    │   ├── StepUpload.jsx       # PDF upload, text preview, worked example
    │   ├── StepChunking.jsx     # 4 strategies + parameters
    │   ├── StepEmbedding.jsx    # 5 models, 2D/3D plots, table view
    │   ├── StepLLM.jsx          # Provider + model selection
    │   ├── StepQuery.jsx        # Query, 3-way comparison, query-time knobs
    │   ├── AgenticTrace.jsx     # Step-by-step decision trace with paper mapping
    │   ├── ChunkGradeCards.jsx  # Per-chunk relevance grades
    │   ├── StepExplainer.jsx    # Collapsible per-step teaching panel
    │   └── InfoPopover.jsx      # Inline concept popovers
    └── utils/
        ├── wizardState.js       # Step completion + downstream invalidation
        └── tracePacing.js       # Replays the trace at real measured latencies

backend/ (FastAPI)
└── app/
    ├── main.py                  # API endpoints, per-tab session state
    ├── pdf_processor.py         # PyMuPDF extraction + page-furniture removal
    ├── chunker.py               # 4 chunking strategies
    ├── embedder.py              # sentence-transformers + UMAP/PCA projection
    ├── retriever.py             # FAISS cosine search
    ├── agentic_rag.py           # Route, grade, verify, reformulate, retry
    ├── llm_handler.py           # Groq + Ollama provider abstraction
    ├── example_data.py          # Pre-chunked worked-example corpus
    └── data/                    # Nutrition textbook excerpt
```

## 🧪 Tests & CI

```bash
# Backend (68 tests: chunking, agentic pipeline, API/sessions, LLM handler, retrieval, PDF cleaning)
pytest backend/tests

# Frontend logic tests + production build
cd frontend && npm test && npm run build
```

CI runs both on every push/PR ([.github/workflows/ci.yml](.github/workflows/ci.yml)).
Retrieval tests that need a real embedding model skip automatically when the model
isn't in the local HuggingFace cache; set `RUN_MODEL_TESTS=1` to force the download.

## ⚠️ Session Model & Limitations

Pipeline state (document, chunks, index, model selection) is scoped **per browser tab** via an
`X-Session-Id` header, so concurrent users on one deployed instance don't interfere with each
other. State is held **in memory in a single process**: it is lost on restart, capped at 50
concurrent sessions (LRU-evicted), and will not work behind multi-worker (`--workers N`) or
multi-replica deployments without sticky routing. For classroom use, one uvicorn process
handles a typical class comfortably; for anything larger, add a shared store (e.g. Redis).

## 🌐 Deployment

| Component | Platform | Cost |
|-----------|----------|------|
| Frontend | Vercel | Free |
| Backend | Railway / Render | ~$7/mo |
| LLM | Groq Cloud API | Free tier |

## 📦 Tech Stack

- **Frontend**: React 19, Vite, Plotly.js, Axios, Lucide Icons
- **Backend**: FastAPI, PyMuPDF, sentence-transformers, FAISS, UMAP, scikit-learn
- **LLM Providers**: Ollama (local), Groq (cloud)
- **Embedding Models**: HuggingFace sentence-transformers

## 📚 Research Background

The **Agentic RAG** mode is a pedagogical implementation of self-correcting RAG inspired by recent literature:

* **Self-RAG**, Asai, Wu, Wang, Sil & Hajishirzi, *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*, 2023. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511), provides the reflection-based retrieval, relevance grading, and answer-support mechanisms.

* **FLARE**, Jiang et al., *Active Retrieval Augmented Generation*, EMNLP 2023. [DOI:10.18653/v1/2023.emnlp-main.495](https://doi.org/10.18653/v1/2023.emnlp-main.495), provides the adaptive retrieval policy underlying the retrieve-vs-answer-directly decision.

* **Search-R1**, Jin et al., *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, 2025. [arXiv:2503.09516](https://arxiv.org/abs/2503.09516), demonstrates learned search decisions and reasoning over retrieved results.

* **RAG**, Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2021. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401), provides the foundational retrieval-augmented generation framework.

* **Agentic RAG Survey**, Singh et al., *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*, 2026. [arXiv:2501.09136](https://arxiv.org/abs/2501.09136), frames reflection, verification, and iterative retrieval as an active design space in agentic RAG systems.

* **CRAG**, Yan, Gu, Zhu & Ling, *Corrective Retrieval Augmented Generation*, 2024. [arXiv:2401.15884](https://arxiv.org/abs/2401.15884), provides the retrieval evaluator and corrective query-reformulation loop.


## Github Repo: [https://github.com/ChandraPrakash-Bathula/AgenticRAG.git]
#HF_Space: [https://huggingface.co/spaces/ChandraPrakashBathula/Self-RAG]

## 📄 License

MIT, see [LICENSE](LICENSE).
