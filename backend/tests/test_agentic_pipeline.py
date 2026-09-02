"""Agentic pipeline behavior with mocked LLM calls: fail-closed grading, original-query
generation, relevance-threshold steering, and grader-model fallback."""
import json

import pytest

from app import agentic_rag


ORIGINAL = "What is the Zorblax-9 protocol?"
REWRITE = "Explain the Zorblax-9 communication handshake specification"


@pytest.fixture
def two_chunks(monkeypatch):
    monkeypatch.setattr(
        agentic_rag, "retrieve_chunks",
        lambda q, i, c, m, top_k=3: [
            {"index": 0, "text": "a", "rank": 1},
            {"index": 1, "text": "b", "rank": 2},
        ],
    )
    monkeypatch.setattr(agentic_rag, "is_model_available", lambda m: True)


def structured_responder(grades_relevant):
    def fake(model_id, system, user, stats_sink=None, call_label="s"):
        if call_label == "route":
            return json.dumps({"needsRetrieval": True, "reason": "r"})
        if call_label == "gradeChunks":
            return json.dumps({"grades": [
                {"chunkId": "0", "relevant": grades_relevant[0]},
                {"chunkId": "1", "relevant": grades_relevant[1]},
            ]})
        if call_label == "gradeAnswer":
            return json.dumps({"supported": True, "missing": "", "confidence": "high"})
        return json.dumps({"newQuery": REWRITE})
    return fake


# ── Fail-closed grading ──────────────────────────────────────────────────────

def test_unparseable_chunk_grades_fail_closed(monkeypatch):
    monkeypatch.setattr(agentic_rag, "query_llm_structured", lambda *a, **k: "not json {")
    grades = agentic_rag.grade_chunks("m", "q", [{"id": "0", "text": "a"}, {"id": "1", "text": "b"}])
    assert all(g["relevant"] is False and g.get("gradingFailed") for g in grades)


def test_unmentioned_chunk_fails_closed(monkeypatch):
    monkeypatch.setattr(
        agentic_rag, "query_llm_structured",
        lambda *a, **k: '{"grades": [{"chunkId": "0", "relevant": true}]}',
    )
    grades = agentic_rag.grade_chunks("m", "q", [{"id": "0", "text": "a"}, {"id": "1", "text": "b"}])
    assert grades[0]["relevant"] is True and grades[1]["relevant"] is False


def test_unparseable_answer_grade_fails_closed(monkeypatch):
    monkeypatch.setattr(agentic_rag, "query_llm_structured", lambda *a, **k: "garbage")
    verdict = agentic_rag.grade_answer("m", "q", "draft", [{"id": "0", "text": "a"}])
    assert verdict["supported"] is False
    assert verdict["confidence"] == "low"
    assert "could not be parsed" in verdict["missing"]


# ── Final answer uses the ORIGINAL question ──────────────────────────────────

def test_generation_uses_original_query_after_reformulation(monkeypatch, two_chunks):
    generation_queries = []
    call_count = {"gradeChunks": 0}

    def fake_structured(model_id, system, user, stats_sink=None, call_label="s"):
        if call_label == "route":
            return json.dumps({"needsRetrieval": True, "reason": "t"})
        if call_label == "gradeChunks":
            call_count["gradeChunks"] += 1
            relevant = call_count["gradeChunks"] > 1  # pass 1 fails, pass 2 succeeds
            return json.dumps({"grades": [
                {"chunkId": "0", "relevant": relevant}, {"chunkId": "1", "relevant": relevant},
            ]})
        if call_label == "gradeAnswer":
            return json.dumps({"supported": True, "missing": "", "confidence": "high"})
        return json.dumps({"newQuery": REWRITE})

    def fake_llm(model_id, question, context=None, stats_sink=None, call_label="g"):
        generation_queries.append(question)
        return "answer"

    monkeypatch.setattr(agentic_rag, "query_llm_structured", fake_structured)
    monkeypatch.setattr(agentic_rag, "query_llm", fake_llm)

    result = agentic_rag.run_agentic_pipeline("big", ORIGINAL, None, ["a", "b"], "em", max_loops=3)

    retrieves = [t for t in result["trace"] if t["step"] == "retrieve"]
    assert retrieves[1]["query"] == REWRITE          # retrieval used the rewrite
    assert generation_queries == [ORIGINAL]          # generation used the original
    draft = next(t for t in result["trace"] if t["step"] == "draftAnswer")
    assert draft["query"] == ORIGINAL and draft.get("retrievalQuery") == REWRITE


# ── Relevance threshold ──────────────────────────────────────────────────────

def test_threshold_steers_reformulation(monkeypatch, two_chunks):
    monkeypatch.setattr(agentic_rag, "query_llm_structured", structured_responder([True, False]))
    monkeypatch.setattr(agentic_rag, "query_llm", lambda *a, **k: "answer")

    lenient = agentic_rag.run_agentic_pipeline("m", "q?", None, ["a", "b"], "em", relevance_threshold=0.4)
    strict = agentic_rag.run_agentic_pipeline("m", "q?", None, ["a", "b"], "em", relevance_threshold=0.9, max_loops=2)

    assert lenient["loopsUsed"] == 1
    assert not any(t["step"] == "reformulate" for t in lenient["trace"])
    assert strict["maxLoopsReached"]
    assert sum(1 for t in strict["trace"] if t["step"] == "reformulate") == 2
    assert lenient["maxLoops"] == 3 and strict["maxLoops"] == 2  # echoed for the UI


def test_threshold_zero_never_accepts_empty_evidence(monkeypatch, two_chunks):
    monkeypatch.setattr(agentic_rag, "query_llm_structured", structured_responder([False, False]))
    monkeypatch.setattr(agentic_rag, "query_llm", lambda *a, **k: "answer")
    res = agentic_rag.run_agentic_pipeline("m", "q?", None, ["a", "b"], "em", relevance_threshold=0.0, max_loops=1)
    assert any(t["step"] == "reformulate" for t in res["trace"])


# ── Grader-model fallback ────────────────────────────────────────────────────

def test_grader_falls_back_to_selected_model(monkeypatch, two_chunks):
    monkeypatch.setattr(agentic_rag, "is_model_available", lambda m: False)
    grading_models = []

    def fake_structured(model_id, system, user, stats_sink=None, call_label="s"):
        grading_models.append(model_id)
        if call_label == "route":
            return json.dumps({"needsRetrieval": True, "reason": "r"})
        if call_label == "gradeChunks":
            return json.dumps({"grades": [{"chunkId": "0", "relevant": True}, {"chunkId": "1", "relevant": True}]})
        return json.dumps({"supported": True, "missing": "", "confidence": "high"})

    monkeypatch.setattr(agentic_rag, "query_llm_structured", fake_structured)
    monkeypatch.setattr(agentic_rag, "query_llm", lambda *a, **k: "answer")

    result = agentic_rag.run_agentic_pipeline("users-model", "q?", None, ["a", "b"], "em")
    assert set(grading_models) == {"users-model"}
    assert "phi4-mini" in result["trace"][0].get("note", "")


# ── Router override regex ────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expect", [
    ("What is the main topic of this document?", True),
    ("Summarize the key findings.", True),
    ("According to the author, what happened?", True),
    ("What is a research paper?", False),
    ("How do I write good text prompts?", False),
])
def test_document_reference_regex(query, expect):
    assert bool(agentic_rag._DOCUMENT_REFERENCE_RE.search(query)) is expect


# ── Router document-awareness ────────────────────────────────────────────────
def test_router_prompt_includes_document_context(monkeypatch):
    """Regression: the router used to see only the bare question, so on-topic subject
    questions ('when do CNNs beat transformers?') were answered directly whenever the
    wording didn't happen to name the document."""
    seen = {}

    def capture(model_id, system, user, stats_sink=None, call_label="s"):
        seen["system"], seen["user"] = system, user
        return json.dumps({"needsRetrieval": True, "reason": "on-topic"})

    monkeypatch.setattr(agentic_rag, "query_llm_structured", capture)
    agentic_rag.route_query(
        "m", "When do CNNs beat transformers?",
        document_context="Filename: cnns.pdf\nOpening excerpt: a survey of CNNs and transformers…",
    )
    assert "cnns.pdf" in seen["user"]
    assert "survey of CNNs" in seen["user"]
    assert "When do CNNs beat transformers?" in seen["user"]
    # The instruction that makes the context actionable must survive prompt edits.
    assert "needsRetrieval=true" in seen["system"]


def test_router_omits_document_block_when_no_document(monkeypatch):
    seen = {}

    def capture(model_id, system, user, stats_sink=None, call_label="s"):
        seen["user"] = user
        return json.dumps({"needsRetrieval": False, "reason": "greeting"})

    monkeypatch.setattr(agentic_rag, "query_llm_structured", capture)
    result = agentic_rag.route_query("m", "Hi there")
    assert "Uploaded document" not in seen["user"]
    assert result["needsRetrieval"] is False


def test_pipeline_threads_document_context_to_router(two_chunks, monkeypatch):
    seen = {}

    def capture(model_id, system, user, stats_sink=None, call_label="s"):
        if call_label == "route":
            seen["user"] = user
            return json.dumps({"needsRetrieval": False, "reason": "generic"})
        return json.dumps({})

    monkeypatch.setattr(agentic_rag, "query_llm_structured", capture)
    monkeypatch.setattr(agentic_rag, "query_llm", lambda *a, **k: "direct answer")
    agentic_rag.run_agentic_pipeline(
        "m", "What is protein?", None, ["a", "b"], "e",
        document_context="Filename: nutrition.pdf\nOpening excerpt: macronutrients…",
    )
    assert "nutrition.pdf" in seen["user"]


# ── ISUSE utility scoring ────────────────────────────────────────────────────
def test_utility_is_parsed_and_clamped(monkeypatch):
    """Models return "4", 4, or 4.0. All must land in 1-5 as an int."""
    for raw, expected in [(5, 5), ("4", 4), (3.0, 3), (9, 5), (0, 1), (-2, 1)]:
        monkeypatch.setattr(
            agentic_rag, "query_llm_structured",
            lambda *a, r=raw, **k: json.dumps(
                {"supported": True, "missing": "", "confidence": "high", "utility": r}),
        )
        assert agentic_rag.grade_answer("m", "q", "a", [{"text": "c"}])["utility"] == expected


def test_utility_fails_closed_on_garbage(monkeypatch):
    """An unreadable score must mean useless, not average. A broken check may never
    flatter the answer, the same rule the support verdict follows."""
    monkeypatch.setattr(
        agentic_rag, "query_llm_structured",
        lambda *a, **k: json.dumps(
            {"supported": True, "missing": "", "confidence": "high", "utility": "quite good"}),
    )
    assert agentic_rag.grade_answer("m", "q", "a", [{"text": "c"}])["utility"] == 1


def test_utility_present_when_whole_grade_unparseable(monkeypatch):
    monkeypatch.setattr(agentic_rag, "query_llm_structured", lambda *a, **k: "not json at all")
    g = agentic_rag.grade_answer("m", "q", "a", [{"text": "c"}])
    assert g["supported"] is False and g["utility"] == agentic_rag.UTILITY_ON_FAILURE


def test_utility_is_independent_of_support(monkeypatch):
    """The teaching case: a correct 'the document does not say' is fully supported
    and nearly useless. The two axes must not collapse into one another."""
    monkeypatch.setattr(
        agentic_rag, "query_llm_structured",
        lambda *a, **k: json.dumps(
            {"supported": True, "missing": "", "confidence": "high", "utility": 1}),
    )
    g = agentic_rag.grade_answer("m", "q", "a", [{"text": "c"}])
    assert g["supported"] is True and g["utility"] == 1


def test_pipeline_surfaces_utility(two_chunks, monkeypatch):
    monkeypatch.setattr(agentic_rag, "query_llm_structured", structured_responder([True, True]))
    monkeypatch.setattr(agentic_rag, "query_llm", lambda *a, **k: "drafted answer")
    r = agentic_rag.run_agentic_pipeline("m", ORIGINAL, None, ["a", "b"], "e")
    assert "utility" in r
    assert any(s["step"] == "gradeAnswer" and "utility" in s for s in r["trace"])


def test_direct_answer_reports_utility_as_unknown(two_chunks, monkeypatch):
    """Nothing was retrieved, so there is no grounded answer to rate."""
    def responder(model_id, system, user, stats_sink=None, call_label="s"):
        if call_label == "route":
            return json.dumps({"needsRetrieval": False, "reason": "generic"})
        return json.dumps({})
    monkeypatch.setattr(agentic_rag, "query_llm_structured", responder)
    monkeypatch.setattr(agentic_rag, "query_llm", lambda *a, **k: "direct")
    assert agentic_rag.run_agentic_pipeline("m", "hi", None, ["a"], "e")["utility"] is None
