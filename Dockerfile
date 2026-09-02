# Hugging Face Space (Docker SDK). A Space exposes ONE port, so this image builds the
# React bundle and serves it from the same FastAPI process that answers /api.
#
# Local development does not use this file: run Vite on 5173 and uvicorn on 8000 as the
# README describes. The static mount in app/main.py only activates when a built bundle
# is present, so the two paths never collide.

# ---- Stage 1: build the frontend ----
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# Same-origin: the Space serves API and UI from one host, so no absolute backend URL.
ENV VITE_API_URL=/api
RUN npm run build

# ---- Stage 2: runtime ----
FROM python:3.11-slim
# Spaces run as uid 1000 and the HF caches must be writable by that user.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first: saves roughly 2.5 GB of CUDA wheels the Space cannot use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user backend/ /app/
COPY --chown=user --from=web /web/dist /frontend/dist

# Bake the default embedding model so a cold Space does not block its first request on
# a ~90 MB download. Done as `user` so the cache lands in a writable directory.
USER user
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Spaces route to 7860 by default.
ENV FRONTEND_DIST=/frontend/dist

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
