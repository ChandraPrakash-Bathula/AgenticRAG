"""LLM handler: provider-honest labels, truncation detection, JSON parsing,
availability checks, and transport-failure semantics."""
import importlib
import json

import pytest
import requests

from app import llm_handler


class FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


@pytest.fixture
def ollama_mode(monkeypatch):
    monkeypatch.setattr(llm_handler, "LLM_PROVIDER", "ollama")
    yield
    importlib.reload(llm_handler)


@pytest.fixture
def groq_mode(monkeypatch):
    monkeypatch.setattr(llm_handler, "LLM_PROVIDER", "groq")
    yield
    importlib.reload(llm_handler)


def test_ollama_labels_match_executed_model(ollama_mode):
    expected = {
        "llama3.2": "Llama 3.2 3B (local)",
        "deepseek-r1:1.5b": "DeepSeek-R1 1.5B (local)",
        "gemma3:4b": "Gemma 3 4B (local)",
        "mistral": "Mistral 7B (local)",
        "phi4-mini": "Phi-4 Mini 3.8B (local)",
    }
    for m in llm_handler.get_available_llms():
        assert m["name"] == expected[m["ollama_id"]]
        for groq_only in ("70B", "120B", "32B", "17B"):
            assert groq_only not in m["name"]


def test_groq_labels_keep_cloud_names(groq_mode):
    names = [m["name"] for m in llm_handler.get_available_llms()]
    assert "GPT-OSS 20B" in names and "GPT-OSS 120B" in names


def test_groq_mode_never_offers_a_model_it_cannot_serve(groq_mode):
    """Regression: Groq retired the Llama chat lineup, but the entries stayed in the
    list for Ollama mode. Offering one in Groq mode hands the user an id that 404s."""
    for m in llm_handler.get_available_llms():
        assert m.get("groq_id"), f"{m['id']} has no Groq endpoint but is offered in Groq mode"
    assert llm_handler.get_fast_model_id() == "openai/gpt-oss-20b"
    assert llm_handler.is_model_available(llm_handler.get_fast_model_id()) is True


def test_local_only_model_rejected_with_actionable_error(groq_mode):
    assert llm_handler.is_model_available("local-phi4-mini") is False
    with pytest.raises(llm_handler.LLMUnavailableError, match="only under Ollama"):
        llm_handler._resolve_groq_model("local-phi4-mini")


def test_ollama_mode_keeps_all_five_local_models(ollama_mode):
    """The retired-on-Groq entries must survive here, llama3.2 and phi4-mini are alive
    locally, and phi4-mini is the agentic grader."""
    ids = {m["ollama_id"] for m in llm_handler.get_available_llms()}
    assert ids == {"llama3.2", "deepseek-r1:1.5b", "gemma3:4b", "mistral", "phi4-mini"}
    assert llm_handler.get_fast_model_id() == "local-phi4-mini"
    assert llm_handler.get_fast_model_hint() == "ollama pull phi4-mini"


def test_is_model_available_checks_ollama_tags(ollama_mode, monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda url, timeout=None: FakeResp(200, {"models": [{"name": "llama3.2:latest"}]}),
    )
    assert llm_handler.is_model_available("local-llama3.2") is True    # -> llama3.2
    assert llm_handler.is_model_available("local-phi4-mini") is False  # -> phi4-mini


def test_missing_ollama_model_raises_unavailable(ollama_mode, monkeypatch):
    monkeypatch.setattr(
        requests, "post",
        lambda url, json=None, timeout=None: FakeResp(404, {"error": "not found"}),
    )
    with pytest.raises(llm_handler.LLMUnavailableError, match="ollama pull phi4-mini"):
        llm_handler.query_llm_structured("local-phi4-mini", "sys", "user")


def test_ollama_connection_error_raises_unavailable(ollama_mode, monkeypatch):
    def dead(url, json=None, timeout=None):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(requests, "post", dead)
    with pytest.raises(llm_handler.LLMUnavailableError, match="ollama serve"):
        llm_handler.query_llm_structured("local-llama3.2", "s", "u")


def test_missing_groq_key_raises_unavailable(groq_mode, monkeypatch):
    monkeypatch.setattr(llm_handler, "GROQ_API_KEY", "")
    with pytest.raises(llm_handler.LLMUnavailableError, match="GROQ_API_KEY"):
        llm_handler.query_llm_structured("openai/gpt-oss-20b", "s", "u")


def test_truncation_flags():
    class Choice:
        finish_reason = "length"

    class Usage:
        prompt_tokens, completion_tokens, total_tokens = 10, 512, 522

    class Completion:
        choices = [Choice()]
        usage = Usage()

    u = {}
    llm_handler._fill_groq_usage(u, Completion())
    assert u["truncated"] is True and u["totalTokens"] == 522

    u2 = {}
    llm_handler._fill_ollama_usage(u2, {"done_reason": "length", "prompt_eval_count": 5, "eval_count": 512})
    assert u2["truncated"] is True

    u3 = {}
    llm_handler._fill_ollama_usage(u3, {"done_reason": "stop", "prompt_eval_count": 5, "eval_count": 50})
    assert "truncated" not in u3


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', {"a": 1}),
    ('Sure! Here is the JSON: {"a": 1} hope that helps', {"a": 1}),
    ("not json at all", None),
    ("", None),
])
def test_safe_json_parse(raw, expected):
    assert llm_handler.safe_json_parse(raw) == expected
