---
title: Naive RAG vs Agentic RAG
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Naive RAG vs Agentic RAG

An interactive playground for self-correcting retrieval augmented generation. Build a
naive RAG pipeline over your own document, then run the same question through a
reflect, verify, retry loop and watch every decision.

## Configuring this Space

Set these under **Settings, Variables and secrets**:

| Name | Kind | Value |
|---|---|---|
| `GROQ_API_KEY` | secret | your key from console.groq.com |
| `LLM_PROVIDER` | variable | `groq` |

Without a key the Space starts but every generation call fails, because a Space cannot
run a local Ollama server. The fully local path documented in the main README works when
you clone the repository and run it yourself.

## What differs from a local clone

- **Cloud models only.** Ollama needs a model server on the same machine, which a Space
  does not provide. Clone the repo for the offline path.
- **Session state is in memory and per browser tab.** A Space that sleeps or restarts
  loses uploaded documents and indexes. Rerun the wizard from Step 1.
- **One process.** Do not add `--workers`; pipeline state is not shared across workers.
- **Your key funds every visitor.** A public Space spends your Groq quota on strangers.
  Consider duplicating it privately, or asking visitors to supply their own key.
- **The full textbook is not bundled here.** The worked example runs from a
  pre-extracted Chapter 3-6 excerpt that ships with the code. The complete 1,208-page
  `Human_Nutrition.pdf` lives in the GitHub repository instead, to keep this image
  small; download it there to follow Levels B and C of the exercises.

## Rename before use

Hugging Face expects the Space README at `README.md`. This repository already has one
for the project itself, so rename this file when you create the Space.
