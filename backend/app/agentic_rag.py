"""
Agentic RAG Pipeline, Self-correcting retrieval with routing, chunk grading,
query reformulation, and answer grading.

This is a pedagogical reduction of reflection-based RAG from the literature:
  - Chunk-relevance and answer-support grading mirror the critique steps of
    Self-RAG: Asai, Wu, Wang, Sil & Hajishirzi, "Self-RAG: Learning to Retrieve,
    Generate, and Critique through Self-Reflection", ICLR 2024 (arXiv:2310.11511).
  - The retrieval-evaluator + query-rewrite loop follows Corrective RAG (CRAG):
    Yan, Gu, Zhu & Ling, "Corrective Retrieval Augmented Generation", 2024
    (arXiv:2401.15884).
  - The retrieve-or-answer-directly routing decision echoes active retrieval:
    Jiang et al., "Active Retrieval Augmented Generation", EMNLP 2023
    (arXiv:2305.06983).

Runs alongside the existing naive RAG path in main.py without altering it.
Each step is an isolated LLM call so the frontend can visualize the trace.
"""

import re
from typing import Dict, List, Optional

from app.llm_handler import (
    query_llm,
    query_llm_structured,
    safe_json_parse,
    is_model_available,
    get_fast_model_id,
    get_fast_model_hint,
)
from app.retriever import retrieve_chunks

RELEVANCE_THRESHOLD = 0.5

# Routing/grading/reformulation are simple classification tasks, always run them on a
# fast, cheap model regardless of which model the user selected for generation. This cuts
# both latency and the shared per-minute rate-limit budget, since agentic mode makes 4-8x
# more calls per query than naive mode. Only actual answer generation uses the user's pick.
# Which model that is depends on the provider (see llm_handler.get_fast_model_id), so it
# must be resolved per call, a module-level constant would freeze one provider's choice.

# Deterministic safety net: if the query explicitly references "the document",
# no LLM routing call should be able to skip retrieval. Catches weak-model
# misroutes on exactly the kind of question this app's UI suggests by default
# (e.g. "What is the main topic of this document?").
# Deliberately narrow: requires a determiner before the noun ("this paper", "the
# uploaded file"), or an explicit sourcing/summarization verb, a bare "text" or
# "paper" in a generic question ("What is a research paper?") must NOT force retrieval.
_DOCUMENT_REFERENCE_RE = re.compile(
    r"\b(?:this|that|the|its?|uploaded|attached)\s+(?:uploaded\s+|attached\s+)?"
    r"(?:document|pdf|paper|text|article|file)\b"
    r"|\baccording to\b"
    r"|\bsummar(?:y|ize|ise)\b",
    re.IGNORECASE,
)


def route_query(
    model_id: str,
    query: str,
    stats: Optional[list] = None,
    document_context: Optional[str] = None,
) -> Dict:
    # Without document_context the router only ever sees the bare question, so it has no
    # way to tell "generic ML question" from "question this specific document answers",
    # it routed subject-matter questions straight past retrieval whenever the wording
    # didn't happen to name the document. Showing it what was uploaded is the fix.
    system = (
        "You are a routing component in a RAG pipeline embedded in a document Q&A app. "
        "The user has already uploaded a PDF and built a retrieval index over it, your job "
        "is to decide whether THIS SPECIFIC query needs to consult that document, or whether "
        "it's a generic query with no relation to any uploaded document (e.g. greetings like "
        "'hi', meta-questions about the assistant itself, simple math like '2+2').\n"
        "You are shown a filename and a few short excerpts sampled from the document. "
        "Those excerpts are a TINY FRACTION of the whole, they show the document's general "
        "subject area, NOT the full list of topics it covers. A question about the same "
        "broad subject is almost certainly answerable from some other part of the document "
        "you were not shown, so judge by SUBJECT AREA, never by whether the excerpts happen "
        "to mention the exact term.\n"
        "If the query relates to the document's subject area at all, needsRetrieval=true, "
        "even when you could answer from general knowledge, and even when the query never "
        "mentions the document. Grounding the answer in the user's own source is the point.\n"
        "Only set needsRetrieval=false for queries that clearly have NOTHING to do with the "
        "subject area: greetings, chit-chat, arithmetic, or questions about you.\n"
        "Examples (assume a document about deep learning architectures):\n"
        '- "What is the main topic of this document?" -> {"needsRetrieval": true, "reason": "Explicitly asks about the document"}\n'
        '- "When do CNNs beat transformers?" -> {"needsRetrieval": true, "reason": "The document covers this subject, ground the answer in it"}\n'
        '- "Summarize the key findings" -> {"needsRetrieval": true, "reason": "Requires document content to summarize"}\n'
        '- "Hi, how are you?" -> {"needsRetrieval": false, "reason": "Greeting, unrelated to the document"}\n'
        '- "What is 12 * 8?" -> {"needsRetrieval": false, "reason": "Simple math, no document needed"}\n'
        'Respond ONLY in JSON: {"needsRetrieval": true|false, "reason": "<one short sentence>"}'
    )
    doc_block = f"Uploaded document:\n{document_context}\n\n" if document_context else ""
    user = f"{doc_block}User query: {query}"
    raw = query_llm_structured(model_id, system, user, stats_sink=stats, call_label="route")
    parsed = safe_json_parse(raw)
    if not isinstance(parsed, dict) or "needsRetrieval" not in parsed:
        result = {"needsRetrieval": True, "reason": "Routing call failed to parse, defaulting to retrieval."}
    else:
        result = {"needsRetrieval": bool(parsed["needsRetrieval"]), "reason": parsed.get("reason", "")}

    if not result["needsRetrieval"] and _DOCUMENT_REFERENCE_RE.search(query):
        result = {"needsRetrieval": True, "reason": "Query explicitly references the document, overriding router."}

    return result


def grade_chunks(model_id: str, query: str, chunks: List[Dict], stats: Optional[list] = None) -> List[Dict]:
    system = (
        "You are a relevance-grading component in a RAG pipeline. You will be given a "
        "user query and a list of retrieved text chunks. For each chunk, decide if it "
        "contains information relevant to answering the query.\n"
        'Respond ONLY in JSON: {"grades": [{"chunkId": "<id>", "relevant": true|false}, ...]}'
    )
    numbered = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    user = f"Query: {query}\nChunks:\n{numbered}"
    raw = query_llm_structured(model_id, system, user, stats_sink=stats, call_label="gradeChunks")
    parsed = safe_json_parse(raw)

    grades = None
    if isinstance(parsed, dict):
        grades = parsed.get("grades")
    elif isinstance(parsed, list):
        grades = parsed

    if not isinstance(grades, list):
        # Fail closed: an unparseable grader response must NOT silently wave chunks
        # through, that would make the pipeline look most confident exactly when its
        # verifier is broken. Treat every chunk as not-relevant and flag the failure
        # so the trace can surface it.
        return [{"chunkId": c["id"], "relevant": False, "gradingFailed": True} for c in chunks]

    by_id = {str(g.get("chunkId")): bool(g.get("relevant")) for g in grades if isinstance(g, dict)}
    # Chunks the grader didn't mention also fail closed.
    return [{"chunkId": c["id"], "relevant": by_id.get(str(c["id"]), False)} for c in chunks]


def reformulate_query(
    model_id: str,
    original_query: str,
    failure_reasons: List[str],
    tried_queries: List[str],
    stats: Optional[list] = None,
) -> str:
    # Always reformulates from the *original* query, not the previous rewrite, chaining
    # rewrite-of-a-rewrite compounds drift across passes and can wander away from what the
    # user actually asked. Accumulated failure reasons + already-tried phrasings give the
    # rewriter context without letting it forget the original intent.
    system = (
        "You are a query-rewriting component in a RAG pipeline. Retrieval attempts based on "
        "the original query (or previous rewrites of it) failed to return enough relevant or "
        "well-supported results. Rewrite the ORIGINAL user query to be more specific, use "
        "different phrasing or synonyms, or break it into a more targeted sub-question, so "
        "retrieval is more likely to succeed. Do not repeat a phrasing already tried.\n"
        'Respond ONLY in JSON: {"newQuery": "<rewritten query>"}'
    )
    reasons_text = "; ".join(failure_reasons) if failure_reasons else "insufficient relevant chunks"
    tried_text = "; ".join(f'"{q}"' for q in tried_queries) if tried_queries else "(none yet)"
    user = (
        f"Original query: {original_query}\n"
        f"Reasons retrieval attempts have been insufficient so far: {reasons_text}\n"
        f"Phrasings already tried (avoid repeating): {tried_text}"
    )
    raw = query_llm_structured(model_id, system, user, stats_sink=stats, call_label="reformulate")
    parsed = safe_json_parse(raw)
    if not isinstance(parsed, dict) or not parsed.get("newQuery"):
        return original_query
    return parsed["newQuery"]


# Self-RAG's ISUSE token, scored 1 to 5. Deliberately a SEPARATE axis from `supported`:
# an answer can be perfectly grounded in the retrieved chunks and still fail to answer
# what was asked ("the document does not say" is fully supported and rarely useful).
# Collapsing the two would hide exactly the case worth teaching.
UTILITY_MIN, UTILITY_MAX = 1, 5
UTILITY_ON_FAILURE = UTILITY_MIN


def _coerce_utility(value) -> int:
    """Clamp the grader's utility score into 1-5, failing closed to the floor.

    Models return "4", 4, 4.0, or prose. Anything unreadable is treated as useless
    rather than average, for the same reason an unparseable support verdict is
    treated as unsupported: a broken check must never flatter the answer."""
    try:
        return max(UTILITY_MIN, min(UTILITY_MAX, int(float(value))))
    except (TypeError, ValueError):
        return UTILITY_ON_FAILURE


def grade_answer(model_id: str, query: str, draft_answer: str, chunks: List[Dict], stats: Optional[list] = None) -> Dict:
    system = (
        "You are a fact-checking component in a RAG pipeline. You will be given a draft "
        "answer and the source chunks it was generated from. Determine whether every claim "
        "in the answer is actually supported by the chunks. Flag anything unsupported or "
        "fabricated.\n"
        "Separately, rate how USEFUL the answer is as a response to the query, from 1 to 5. "
        "These are different judgements: an answer that correctly says the document does not "
        "cover the question is fully supported but has low utility. 5 means it directly and "
        "completely answers what was asked; 1 means it does not address the query at all.\n"
        'Respond ONLY in JSON: {"supported": true|false, "missing": "<what\'s unsupported or '
        'missing, empty string if none>", "confidence": "high"|"medium"|"low", "utility": 1-5}'
    )
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    user = f"Query: {query}\nDraft answer: {draft_answer}\nSource chunks:\n{context}"
    raw = query_llm_structured(model_id, system, user, stats_sink=stats, call_label="gradeAnswer")
    parsed = safe_json_parse(raw)
    if not isinstance(parsed, dict) or "supported" not in parsed:
        # Fail closed: if the fact-checker's response can't be parsed, the answer is
        # unverified, say so instead of rubber-stamping it as supported.
        return {
            "supported": False,
            "missing": "grader response could not be parsed, defaulting to unsupported",
            "confidence": "low",
            "utility": UTILITY_ON_FAILURE,
            "gradingFailed": True,
        }
    return {
        "supported": bool(parsed["supported"]),
        "missing": parsed.get("missing", ""),
        "confidence": parsed.get("confidence", "medium"),
        "utility": _coerce_utility(parsed.get("utility")),
    }


def generate_direct_answer(model_id: str, query: str, stats: Optional[list] = None) -> str:
    return query_llm(model_id, query, context=None, stats_sink=stats, call_label="directAnswer")


def generate_answer(model_id: str, query: str, chunks: List[Dict], stats: Optional[list] = None) -> str:
    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    return query_llm(model_id, query, context=context, stats_sink=stats, call_label="draftAnswer")


def _aggregate_stats(stats: list) -> Dict:
    total_tokens = [s["totalTokens"] for s in stats if s.get("totalTokens") is not None]
    return {
        "calls": len(stats),
        "latencyMs": round(sum(s["elapsedMs"] for s in stats), 1),
        "totalTokens": sum(total_tokens) if total_tokens else None,
        "truncated": any(s.get("truncated") for s in stats),
        "calls_detail": stats,
    }


def run_agentic_pipeline(
    model_id: str,
    query: str,
    faiss_index,
    chunks: List[str],
    embedding_model: str,
    top_k: int = 3,
    max_loops: int = 3,
    relevance_threshold: float = RELEVANCE_THRESHOLD,
    demo_mode: bool = False,
    document_context: Optional[str] = None,
) -> Dict:
    trace = []
    step_id = 0
    stats: list = []

    def log(step: str, loop: int, **data) -> None:
        nonlocal step_id
        trace.append({"stepId": step_id, "step": step, "loop": loop, **data})
        step_id += 1

    # Resolve the grading/routing model up front. The fast model is preferred, but if
    # it isn't available (e.g. Ollama user never pulled phi4-mini), fall back to the
    # user's selected model rather than letting every structured call fail, and say
    # so in the trace instead of degrading silently.
    fast_model = get_fast_model_id()
    grader_model = fast_model
    grader_fallback = None
    if not is_model_available(fast_model):
        grader_model = model_id
        grader_fallback = (
            "Fast grader model isn't available locally, using your selected model for "
            f"routing/grading/reformulation instead (slower). Install it with: {get_fast_model_hint()}"
        )

    route = route_query(grader_model, query, stats=stats, document_context=document_context)
    log("route", 0, **route, **({"note": grader_fallback} if grader_fallback else {}))

    if not route["needsRetrieval"]:
        answer = generate_direct_answer(model_id, query, stats=stats)
        log("directAnswer", 0, answer=answer)
        return {
            "answer": answer,
            "confidence": "high",
            "utility": None,  # nothing was retrieved, so there is no grounded answer to rate
            "trace": trace,
            "retrievedChunks": [],
            "loopsUsed": 0,
            "maxLoops": max_loops,
            "maxLoopsReached": False,
            "stats": _aggregate_stats(stats),
        }

    current_query = query
    tried_queries = [query]
    failure_reasons: List[str] = []
    best_attempt: Dict = {"answer": None, "confidence": "low",
                          "utility": UTILITY_ON_FAILURE, "chunks": []}
    last_retrieved_chunks: List[Dict] = []
    loop_count = 0

    while loop_count < max_loops:
        retrieved = retrieve_chunks(current_query, faiss_index, chunks, embedding_model, top_k=top_k)
        graded_chunks = [{"id": str(r["index"]), "text": r["text"], "rank": r["rank"]} for r in retrieved]
        last_retrieved_chunks = graded_chunks
        log("retrieve", loop_count, query=current_query, chunks=graded_chunks)

        if demo_mode and loop_count == 0:
            # Deliberately staged for demo reliability, every subsequent step (including
            # pass-2 grading) runs for real. Clearly flagged in the trace, not hidden.
            grades = [{"chunkId": c["id"], "relevant": False} for c in graded_chunks]
            log(
                "gradeChunks",
                loop_count,
                query=current_query,
                grades=grades,
                demo=True,
                note="🎬 Demo mode: pass 1 grading intentionally forced insufficient to reliably show the self-correction loop.",
            )
        else:
            grades = grade_chunks(grader_model, current_query, graded_chunks, stats=stats)
            grading_failed = any(g.get("gradingFailed") for g in grades)
            extra = (
                {"note": "grader response could not be parsed, defaulting to not relevant (fail-closed)"}
                if grading_failed
                else {}
            )
            log("gradeChunks", loop_count, query=current_query, grades=grades, **extra)

        relevant_ids = {g["chunkId"] for g in grades if g["relevant"]}
        relevant_chunks = [c for c in graded_chunks if c["id"] in relevant_ids]

        # Zero relevant chunks always reformulates, even at threshold 0, an empty
        # evidence set must never be "accepted" and drafted from.
        if (
            not graded_chunks
            or not relevant_chunks
            or len(relevant_chunks) / len(graded_chunks) < relevance_threshold
        ):
            failure_reasons.append("insufficient relevant chunks")
            new_query = reformulate_query(grader_model, query, failure_reasons, tried_queries, stats=stats)
            tried_queries.append(new_query)
            log("reformulate", loop_count, query=current_query, reason="insufficient relevant chunks", newQuery=new_query)
            current_query = new_query
            loop_count += 1
            continue

        # Reformulated queries exist to steer RETRIEVAL only. The answer itself must
        # always address the user's ORIGINAL question, otherwise reformulation drift
        # produces a confident answer to a question nobody asked.
        draft_answer = generate_answer(model_id, query, relevant_chunks, stats=stats)
        drift = (
            {"retrievalQuery": current_query, "note": "Chunks were retrieved via a reformulated query; the answer addresses the original question."}
            if current_query != query
            else {}
        )
        log("draftAnswer", loop_count, query=query, draftAnswer=draft_answer, **drift)

        grade = grade_answer(grader_model, query, draft_answer, relevant_chunks, stats=stats)
        log("gradeAnswer", loop_count, query=query, **grade)

        best_attempt = {"answer": draft_answer, "confidence": grade["confidence"],
                        "utility": grade["utility"], "chunks": relevant_chunks}

        if grade["supported"]:
            return {
                "answer": draft_answer,
                "confidence": grade["confidence"],
                "utility": grade["utility"],
                "trace": trace,
                "retrievedChunks": relevant_chunks,
                "loopsUsed": loop_count + 1,
                "maxLoops": max_loops,
                "maxLoopsReached": False,
                "stats": _aggregate_stats(stats),
            }

        reason = grade["missing"] or "answer not fully supported"
        failure_reasons.append(reason)
        new_query = reformulate_query(grader_model, query, failure_reasons, tried_queries, stats=stats)
        tried_queries.append(new_query)
        log("reformulate", loop_count, query=current_query, reason=reason, newQuery=new_query)
        current_query = new_query
        loop_count += 1

    # By construction: reaching this point means every gradeAnswer call along the way (if
    # any) returned supported=False, a supported=True verdict always returns early, above.
    # So "a real draft was produced" and "grade_answer caught something unsupported" are the
    # same condition here, and distinguishing them from "no draft ever survived chunk-grading"
    # tells us WHY max-loops was hit, not just that it was.
    had_unsupported_draft = best_attempt["answer"] is not None

    if best_attempt["answer"] is None:
        # No pass ever produced a draft (every loop failed at chunk-grading). Ground the
        # fallback in whatever was actually retrieved on the last pass, even though it was
        # graded insufficient, rather than asking the model with zero context, which produces
        # an incoherent "there's nothing here" non-answer instead of an honest low-confidence one.
        if last_retrieved_chunks:
            best_attempt["answer"] = generate_answer(model_id, query, last_retrieved_chunks, stats=stats)
            best_attempt["chunks"] = last_retrieved_chunks
            fallback_note = "Fallback: grounded in the last retrieved chunks, though grading found them insufficient."
        else:
            best_attempt["answer"] = generate_direct_answer(model_id, query, stats=stats)
            fallback_note = "Fallback: no chunks were ever retrieved, answered without document context."
        log("draftAnswer", loop_count, query=query, draftAnswer=best_attempt["answer"], note=fallback_note)

    max_loops_reason = "answer_unsupported" if had_unsupported_draft else "chunk_grading_noise"

    return {
        "answer": best_attempt["answer"],
        "confidence": "low",
        "utility": best_attempt["utility"],
        "note": "Max loops reached, returning best attempt.",
        "trace": trace,
        "retrievedChunks": best_attempt["chunks"],
        "loopsUsed": loop_count,
        "maxLoops": max_loops,
        "maxLoopsReached": True,
        "maxLoopsReason": max_loops_reason,
        "stats": _aggregate_stats(stats),
    }
