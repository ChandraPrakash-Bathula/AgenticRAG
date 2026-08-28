"""
LLM Handler, Supports both Ollama (local) and Groq (cloud) inference.

Set LLM_PROVIDER env var:
  - "ollama" → uses local Ollama server (default for dev)
  - "groq"   → uses Groq cloud API (recommended for deployment)

For Groq, set GROQ_API_KEY env var.
"""

import os
import re
import json
import time
import requests
from typing import Optional


class LLMRateLimitError(RuntimeError):
    """Raised when a provider call fails due to rate limiting and could not be recovered
    by the SDK's built-in retry. Callers must not treat this as a normal text response,
    a rate-limit failure silently returned as a plain string can flow into a downstream
    grading call and get rubber-stamped as a "verified" answer."""


class LLMUnavailableError(RuntimeError):
    """Raised when the provider/model cannot be reached at all (model not pulled,
    Ollama not running, API key missing). Distinct from a parse failure on a response
    the model DID produce: parse failures fail closed inside the pipeline, whereas an
    unavailable model must abort with an actionable message, otherwise a missing
    `ollama pull` masquerades as 'the grader found nothing relevant'."""

# ── Fix SSL cert path (same issue as embedder.py) ────────────────────────────
_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.exists(_ssl_cert):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        os.environ.pop("SSL_CERT_FILE", None)

# ── Configuration ─────────────────────────────────────────────────────────────
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")  # "ollama" or "groq"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# One shared Groq client (created lazily) with a request timeout, instead of a new
# client per call with no timeout, a stuck call must not hang a worker indefinitely.
GROQ_TIMEOUT_SECONDS = 30.0
_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY, timeout=GROQ_TIMEOUT_SECONDS)
    return _groq_client

# ── Model definitions ────────────────────────────────────────────────────────
# Each entry pairs a Groq cloud model with the DIFFERENT (smaller) Ollama model that
# runs locally in its place. The UI must always display the model that actually
# executes, never advertise the Groq name while running the local substitute.
#
# `groq_id: None` marks a LOCAL-ONLY entry: Groq retired Meta's entire Llama chat
# lineup (llama-3.3-70b-versatile and llama-3.1-8b-instant both 404 as of 2026-08;
# only openai/gpt-oss-* and qwen/qwen3.6-27b survive as general chat models on their
# catalog). The local counterparts, llama3.2 and phi4-mini, are alive and well in
# Ollama, so the entries stay and get filtered out of Groq mode instead of deleted.
# Their `id` is a provider-neutral slug precisely because no Groq id backs them.
#
# `ollama_id: None` is the mirror image: a CLOUD-ONLY entry with no local counterpart
# worth shipping. Each provider's list is therefore a filtered view of this one table.
AVAILABLE_LLMS = [
    {
        # The only small chat model left on Groq's catalog, the entire Llama 3.x line
        # is gone, so this is what carries the "small model + RAG beats big model alone"
        # lesson in cloud mode. Note the 4K context (vs 131K for every other entry):
        # CONTEXT_LIMITS below keeps retrieved chunks from overflowing it.
        "id": "allam-2-7b",
        "explainer": "SDAIA's Arabic-English bilingual model (7 billion parameters). The smallest chat model Groq still serves, useful for watching how much of the answer quality comes from retrieval rather than model scale. Its 4K context window is 32x smaller than the other cloud models here, so it holds far fewer retrieved chunks.",
        "ollama_explainer": None,
        "ollama_id": None,
        "groq_id": "allam-2-7b",
        "name": "Allam 2 7B",
        "description": "Smallest cloud model. 7B parameters, 4K context, a real small-model baseline.",
        "size": "7B",
        "ollama_name": None,
        "ollama_description": None,
        "ollama_size": None,
        "icon": "🪶",
    },
    {
        # Local-only since Groq retired llama-3.3-70b-versatile (see header note).
        "id": "local-llama3.2",
        "explainer": "Meta's flagship open-weights chat model. No longer served by Groq, this entry now runs only under Ollama.",
        "ollama_explainer": "Meta's compact on-device model (2024): 3 billion parameters. Fast on a laptop CPU and solid at grounded Q&A, where the retrieved context does the heavy lifting, a good demonstration that RAG lets small models punch above their weight.",
        "ollama_id": "llama3.2",
        "groq_id": None,
        "name": "Llama 3.2 3B (local)",
        "description": "Meta's most capable open model. Excellent across all tasks.",
        "size": "70B",
        "ollama_name": "Llama 3.2 3B (local)",
        "ollama_description": "Meta's compact local model. Fast on CPU, solid general quality.",
        "ollama_size": "3B",
        "icon": "🦙",
    },
    {
        # Groq retired the Llama 4 Scout endpoint (meta-llama/llama-4-scout-17b-16e-instruct
        # 404s as of 2026-07); GPT-OSS 20B is OpenAI's smaller open-weight sibling of the
        # 120B entry below and is live on Groq's current model catalog.
        "id": "openai/gpt-oss-20b",
        "explainer": "OpenAI's smaller open-weight mixture-of-experts model (2025): ~21B total parameters, a fraction active per token via MoE routing. A lighter, faster sibling of GPT-OSS 120B below, useful for comparing how much reasoning quality that extra scale actually buys.",
        "ollama_explainer": "A tiny (1.5B) model distilled from DeepSeek-R1's reasoning traces: it 'thinks out loud' before answering. Fun to watch in the trace, but limited capacity, expect it to struggle with long contexts.",
        "ollama_id": "deepseek-r1:1.5b",
        "groq_id": "openai/gpt-oss-20b",
        "name": "GPT-OSS 20B",
        "description": "OpenAI's smaller open-weight model. Faster than the 120B, still solid reasoning.",
        "size": "20B",
        "ollama_name": "DeepSeek-R1 1.5B (local)",
        "ollama_description": "DeepSeek's tiny reasoning-distilled model. Very fast, thinks out loud.",
        "ollama_size": "1.5B",
        "icon": "🔍",
    },
    {
        # Groq renamed/upgraded qwen/qwen3-32b to qwen/qwen3.6-27b; the old id 404s.
        "id": "qwen/qwen3.6-27b",
        "explainer": "Alibaba's Qwen 3.6 dense 27B model with an optional extended-thinking mode. Strong multilingual coverage and reasoning, a good choice for non-English documents.",
        "ollama_explainer": "Google's small open model from the Gemma 3 family (4B parameters). Good quality-per-parameter for local use on modest hardware.",
        "ollama_id": "gemma3:4b",
        "groq_id": "qwen/qwen3.6-27b",
        "name": "Qwen 3.6 27B",
        "description": "Alibaba's powerful reasoning model. Excellent multilingual support.",
        "size": "27B",
        "ollama_name": "Gemma 3 4B (local)",
        "ollama_description": "Google's small open model. Good quality for its size.",
        "ollama_size": "4B",
        "icon": "🧮",
    },
    {
        "id": "openai/gpt-oss-120b",
        "explainer": "OpenAI's open-weight mixture-of-experts model (2025): ~117B total parameters with ~5B active per token. The largest option here; strong reasoning, still fast on Groq thanks to the MoE design.",
        "ollama_explainer": "Mistral's classic dense 7B model, the 2023 release that proved small open models could compete. Still a reliable local workhorse for grounded answering.",
        "ollama_id": "mistral",
        "groq_id": "openai/gpt-oss-120b",
        "name": "GPT-OSS 120B",
        "description": "OpenAI's open-weight model. Largest available, top quality.",
        "size": "120B",
        "ollama_name": "Mistral 7B (local)",
        "ollama_description": "Mistral's classic 7B model. Reliable local workhorse.",
        "ollama_size": "7B",
        "icon": "🌟",
    },
    {
        # Local-only since Groq retired llama-3.1-8b-instant (see header note).
        "id": "local-phi4-mini",
        "explainer": "Meta's 8B speed-tuned model. No longer served by Groq, this entry now runs only under Ollama.",
        "ollama_explainer": "Microsoft's Phi-4 Mini (3.8B), trained on heavily curated 'textbook-quality' data. Doubles as the local grader for agentic mode's routing/grading calls.",
        "ollama_id": "phi4-mini",
        "groq_id": None,
        "name": "Phi-4 Mini 3.8B (local)",
        "description": "Ultra-fast lightweight model. Best for quick responses.",
        "size": "8B",
        "ollama_name": "Phi-4 Mini 3.8B (local)",
        "ollama_description": "Microsoft's small model. Quick responses on modest hardware.",
        "ollama_size": "3.8B",
        "icon": "⚡",
    },
]

# Build lookup dict
_MODEL_MAP = {m["id"]: m for m in AVAILABLE_LLMS}

# Context window per model, in tokens. Only models that differ from the 131K norm need
# an entry. Allam's 4K window is small enough that a few large retrieved chunks plus the
# system prompt can overflow it, Groq answers that with a 413, which would surface to a
# student as an unexplained failure right after they picked the interesting small model.
_DEFAULT_CONTEXT_TOKENS = 131072
CONTEXT_LIMITS = {"allam-2-7b": 4096}


def get_context_limit(model_id: str) -> int:
    """Context window (tokens) for a model id, defaulting to the 131K most models have."""
    groq_id = (_MODEL_MAP.get(model_id) or {}).get("groq_id") or model_id
    return CONTEXT_LIMITS.get(groq_id, _DEFAULT_CONTEXT_TOKENS)


def fit_context_to_model(model_id: str, context: str, reserve_tokens: int = 900) -> str:
    """Trim retrieved context so prompt + completion fit the model's window.

    Uses ~4 chars/token, the standard rough estimate, deliberately approximate, since
    the cost of being wrong is a trimmed chunk, not a failed request. `reserve_tokens`
    covers the system prompt, the question, and the 512-token completion budget."""
    budget_chars = max(0, (get_context_limit(model_id) - reserve_tokens)) * 4
    if len(context) <= budget_chars:
        return context
    return context[:budget_chars].rsplit(" ", 1)[0] + "\n\n[… context truncated to fit this model's window]"

# Routing/grading/reformulation always run on the cheapest model the ACTIVE provider
# can serve, never the user's (possibly large) generation pick. Groq's cheapest live
# chat model is GPT-OSS 20B; Ollama's is Phi-4 Mini. Resolved at call time, not import
# time, so a provider switch (or a test monkeypatching LLM_PROVIDER) is picked up.
_FAST_MODEL_BY_PROVIDER = {"groq": "openai/gpt-oss-20b", "ollama": "local-phi4-mini"}


def get_fast_model_id() -> str:
    """Model id used for agentic routing/grading/reformulation on the active provider."""
    return _FAST_MODEL_BY_PROVIDER["groq" if LLM_PROVIDER == "groq" else "ollama"]


def get_fast_model_hint() -> str:
    """Actionable install hint when the fast grader isn't available (Ollama only)."""
    return f"ollama pull {_MODEL_MAP[get_fast_model_id()]['ollama_id']}"

# Groq "reasoning" models emit hidden chain-of-thought before the final answer and,
# left unchecked, that reasoning both eats into max_completion_tokens (observed: GPT-OSS
# 20B and Qwen3.6 27B burning the ENTIRE 512-1500 token budget on reasoning for a
# no-context question and returning empty, truncated content, reasoning_format=hidden
# alone does not shrink the reasoning budget, only suppresses it from the output) and,
# for Qwen3 specifically, leaks a raw <think>...</think> block into .content when not
# hidden. reasoning_format="hidden" plus a low reasoning_effort fixes both, but Groq
# 400s if either is sent to a non-reasoning model (e.g. the Llama models) or if
# reasoning_effort uses the wrong model's enum (gpt-oss: low/medium/high; qwen3: none/
# default), so both must be added per-model, never unconditionally.
_GROQ_REASONING_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.6-27b"}
_GROQ_REASONING_EFFORT = {
    "openai/gpt-oss-20b": "low",
    "openai/gpt-oss-120b": "low",
    "qwen/qwen3.6-27b": "none",
}


def _resolve_groq_model(model_id: str) -> str:
    """Map a catalog id to the id Groq actually serves.

    Guards the local-only entries: "groq_id" is explicitly None for those, and
    dict.get returns that None rather than the default, sending it to the SDK would
    surface as an opaque validation error instead of an actionable message."""
    info = _MODEL_MAP.get(model_id)
    if info is None:
        return model_id  # unknown id, let Groq reject it by name
    groq_model = info.get("groq_id")
    if not groq_model:
        raise LLMUnavailableError(
            f"{info['name']} is not available on Groq, it runs only under Ollama. "
            f"Pick one of: {', '.join(m['name'] for m in AVAILABLE_LLMS if m.get('groq_id'))}"
        )
    return groq_model


def _groq_reasoning_kwargs(groq_model: str) -> dict:
    if groq_model not in _GROQ_REASONING_MODELS:
        return {}
    return {"reasoning_format": "hidden", "reasoning_effort": _GROQ_REASONING_EFFORT[groq_model]}


def get_available_llms() -> list:
    """Return the model list with name/size/description resolved to the ACTIVE provider,
    so the UI shows exactly the model that will run (e.g. 'Mistral 7B (local)' in
    Ollama mode, never 'GPT-OSS 120B' while a 7B model executes).

    Also drops entries the active provider cannot serve at all, see the AVAILABLE_LLMS
    header on local-only models."""
    resolved = []
    for m in AVAILABLE_LLMS:
        # Local-only entries have no Groq endpoint, offering them in Groq mode would
        # hand the user a model id that 404s the moment they run a query.
        if LLM_PROVIDER == "groq" and not m.get("groq_id"):
            continue
        if LLM_PROVIDER != "groq" and not m.get("ollama_id"):
            continue
        entry = {k: v for k, v in m.items() if not k.startswith("ollama_") or k == "ollama_id"}
        if LLM_PROVIDER != "groq":
            entry["name"] = m["ollama_name"]
            entry["description"] = m["ollama_description"]
            entry["size"] = m["ollama_size"]
            entry["explainer"] = m["ollama_explainer"]
        resolved.append(entry)
    return resolved


def get_provider_info() -> dict:
    """Return current provider configuration."""
    return {
        "provider": LLM_PROVIDER,
        "groq_configured": bool(GROQ_API_KEY),
    }


def check_ollama_status() -> dict:
    """Check if Ollama is running and which models are available."""
    if LLM_PROVIDER == "groq":
        return {
            "running": True,
            "provider": "groq",
            "installed_models": [m["id"] for m in AVAILABLE_LLMS if m.get("groq_id")],
            "message": "Using Groq Cloud API, all models available instantly ⚡",
        }

    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            installed = [m["name"].split(":")[0] for m in models]
            return {"running": True, "provider": "ollama", "installed_models": installed}
    except Exception:
        pass
    return {"running": False, "provider": "ollama", "installed_models": []}


def is_model_available(model_id: str) -> bool:
    """True if the active provider can serve this model right now.
    Groq: only models with a live groq_id (local-only entries 404). Ollama: only if
    the mapped local model is actually pulled and the server is running."""
    info = _MODEL_MAP.get(model_id, {})
    if LLM_PROVIDER == "groq":
        return bool(info.get("groq_id"))
    ollama_model = info.get("ollama_id", model_id) if info else model_id
    if info and not ollama_model:
        return False  # cloud-only entry, no local counterpart
    status = check_ollama_status()
    if not status.get("running"):
        return False
    base = ollama_model.split(":")[0]
    return base in status.get("installed_models", [])


def query_llm(
    model_id: str,
    question: str,
    context: Optional[str] = None,
    stats_sink: Optional[list] = None,
    call_label: str = "generate",
) -> str:
    """
    Query an LLM via the configured provider (Ollama or Groq).

    If stats_sink is provided, appends a {label, model, elapsedMs, promptTokens,
    completionTokens, totalTokens} entry to it, used to show real latency/cost
    numbers in the UI. Passing None (the default) is a no-op, so existing callers
    are unaffected.
    """
    # Trim before dispatch, not inside a provider branch: a small-window model is a
    # property of the model, not of who serves it.
    if context:
        context = fit_context_to_model(model_id, context)
    start = time.perf_counter()
    usage: dict = {}
    if LLM_PROVIDER == "groq":
        text = _query_groq(model_id, question, context, usage)
    else:
        text = _query_ollama(model_id, question, context, usage)
    _record_stats(stats_sink, call_label, model_id, start, usage)
    return text


# ── Structured (JSON) calls, used by the agentic RAG pipeline ──────────────
def query_llm_structured(
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    stats_sink: Optional[list] = None,
    call_label: str = "structured",
) -> str:
    """
    Query an LLM with an explicit system/user prompt pair, requesting JSON output.
    Used by the agentic pipeline's small grading/routing/reformulation calls.
    Returns the raw text response, callers parse it with safe_json_parse.
    """
    start = time.perf_counter()
    usage: dict = {}
    if LLM_PROVIDER == "groq":
        text = _query_groq_structured(model_id, system_prompt, user_prompt, usage)
    else:
        text = _query_ollama_structured(model_id, system_prompt, user_prompt, usage)
    _record_stats(stats_sink, call_label, model_id, start, usage)
    return text


def _record_stats(stats_sink, label, model_id, start_time, usage):
    if stats_sink is None:
        return
    entry = {
        "label": label,
        "model": model_id,
        "elapsedMs": round((time.perf_counter() - start_time) * 1000, 1),
    }
    entry.update(usage)
    stats_sink.append(entry)


def safe_json_parse(text: str):
    """Parse JSON from an LLM response, tolerating extra prose around it."""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None


def _query_groq_structured(model_id: str, system_prompt: str, user_prompt: str, usage_out: Optional[dict] = None) -> str:
    if not GROQ_API_KEY:
        # Config failure, not a model response, raise so it can't be mistaken
        # for a parse failure and silently fail-closed.
        raise LLMUnavailableError(
            "GROQ_API_KEY is not set, get a free key at https://console.groq.com"
        )

    groq_model = _resolve_groq_model(model_id)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    from groq import RateLimitError
    client = _get_groq_client()
    extra = _groq_reasoning_kwargs(groq_model)
    try:
        try:
            completion = client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=512,
                response_format={"type": "json_object"},
                **extra,
            )
        except RateLimitError:
            raise
        except Exception:
            # Some Groq models reject response_format, fall back to a plain call.
            completion = client.chat.completions.create(
                model=groq_model,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=512,
                **extra,
            )
        _fill_groq_usage(usage_out, completion)
        return completion.choices[0].message.content
    except RateLimitError as e:
        # Do not swallow this as an empty string, an empty response reads as a JSON
        # parse failure to callers, which is the wrong failure mode (they fall back to
        # a permissive default like "supported: true" instead of aborting the pipeline).
        raise LLMRateLimitError("Groq rate limit reached, please wait a moment and try again.") from e
    except Exception:
        return ""


def _query_ollama_structured(model_id: str, system_prompt: str, user_prompt: str, usage_out: Optional[dict] = None) -> str:
    model_info = _MODEL_MAP.get(model_id, {})
    ollama_model = model_info.get("ollama_id", model_id)
    prompt = f"{system_prompt}\n\n{user_prompt}"

    # Transport/availability failures RAISE. Only a response the model actually
    # produced may return as a string, so a missing model can never be mistaken
    # for "the grader responded and graded everything irrelevant".
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )
    except requests.ConnectionError as e:
        raise LLMUnavailableError(
            "Cannot connect to Ollama, make sure it is running (ollama serve)."
        ) from e
    except requests.Timeout as e:
        raise LLMUnavailableError(
            f"Ollama request for '{ollama_model}' timed out, the model may still be loading."
        ) from e

    if resp.status_code != 200:
        raise LLMUnavailableError(
            f"Ollama returned HTTP {resp.status_code} for model '{ollama_model}', "
            f"is it installed? Run: ollama pull {ollama_model}"
        )

    data = resp.json()
    _fill_ollama_usage(usage_out, data)
    return data.get("response", "")


def _fill_groq_usage(usage_out: Optional[dict], completion) -> None:
    if usage_out is None:
        return
    choices = getattr(completion, "choices", None) or []
    if choices and getattr(choices[0], "finish_reason", None) == "length":
        usage_out["truncated"] = True
    usage = getattr(completion, "usage", None)
    if usage is None:
        return
    usage_out["promptTokens"] = getattr(usage, "prompt_tokens", None)
    usage_out["completionTokens"] = getattr(usage, "completion_tokens", None)
    usage_out["totalTokens"] = getattr(usage, "total_tokens", None)


def _fill_ollama_usage(usage_out: Optional[dict], data: dict) -> None:
    if usage_out is None:
        return
    if data.get("done_reason") == "length":
        usage_out["truncated"] = True
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    usage_out["promptTokens"] = prompt_tokens
    usage_out["completionTokens"] = completion_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        usage_out["totalTokens"] = prompt_tokens + completion_tokens


# ── Groq Implementation ──────────────────────────────────────────────────────
def _query_groq(model_id: str, question: str, context: Optional[str] = None, usage_out: Optional[dict] = None) -> str:
    """Query via Groq Cloud API."""
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not set. Get a free key at https://console.groq.com"

    groq_model = _resolve_groq_model(model_id)

    messages = []
    if context:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. If the context doesn't contain enough information, say so."
        })
        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })
    else:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Answer the following question to the best of your knowledge."
        })
        messages.append({
            "role": "user",
            "content": question
        })

    from groq import RateLimitError
    client = _get_groq_client()
    extra = _groq_reasoning_kwargs(groq_model)
    try:
        completion = client.chat.completions.create(
            model=groq_model,
            messages=messages,
            temperature=0.3,
            max_completion_tokens=512,
            **extra,
        )
        _fill_groq_usage(usage_out, completion)
        return completion.choices[0].message.content

    except RateLimitError as e:
        # Previously this returned the string "Error: Groq rate limit reached..." as if it
        # were a real answer. In the agentic pipeline that string then got passed to
        # grade_answer() as a "draft answer", and if that grading call *also* got rate
        # limited, its own parse-failure default (supported=True) rubber-stamped the broken
        # string as a verified, medium-confidence answer. Raising here forces callers to
        # handle the failure explicitly instead of silently faking a response.
        raise LLMRateLimitError("Groq rate limit reached, please wait a moment and try again.") from e
    except Exception as e:
        return f"Error: {str(e)}"


# ── Ollama Implementation ────────────────────────────────────────────────────
def _query_ollama(model_id: str, question: str, context: Optional[str] = None, usage_out: Optional[dict] = None) -> str:
    """Query via local Ollama server."""
    model_info = _MODEL_MAP.get(model_id, {})
    ollama_model = model_info.get("ollama_id", model_id)

    if context:
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"""You are a helpful assistant. Answer the following question to the best of your knowledge.

Question: {question}

Answer:"""

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )

        if resp.status_code == 200:
            data = resp.json()
            _fill_ollama_usage(usage_out, data)
            return data.get("response", "No response generated.")
        else:
            return f"Error: Ollama returned status {resp.status_code}. Run: ollama pull {ollama_model}"

    except requests.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)."
    except requests.Timeout:
        return "Error: LLM request timed out. The model might be loading for the first time."
    except Exception as e:
        return f"Error: {str(e)}"
