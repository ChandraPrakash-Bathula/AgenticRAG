"""Generate the 2-page NeurIPS Educational Resources concept PDF.

Usage:  python3 submission/generate_concept_pdf.py
Output: submission/rag-pipeline-studio-concept.pdf (exactly 2 pages, A4)
"""

import os
import pymupdf as fitz

HTML = """
<h1>Self-Correcting RAG, Interactively: Teaching Reflective Retrieval
(Self-RAG / CRAG) by Contrast with the Naive Pipeline It Fixes</h1>
<p class="meta">Chandra Prakash Bathula &middot; chandu.bathula1617@gmail.com &middot;
NeurIPS 2026 Educational Resources Track &middot; Interactive web application + exercises (full source included)</p>

<h2>1&nbsp;&nbsp;The Concept: Reflective, Self-Correcting Retrieval</h2>
<p>The emerging concept this resource teaches is <b>self-corrective retrieval-augmented
generation</b>: pipelines in which the model <i>critiques its own retrieval and generation</i>,
deciding whether to retrieve at all, grading retrieved evidence for relevance, verifying that a
drafted answer is supported by that evidence, and rewriting the query and retrying when a check
fails. Introduced by Self-RAG's reflection tokens [1] and CRAG's retrieval evaluator with
corrective query rewriting [2], with adaptive retrieval policies from FLARE [3], this
reflect-verify-retry loop has become a default ingredient of current agentic search systems:
2025&ndash;2026 work trains exactly these routing/verification behaviors with reinforcement
learning (e.g. Search-R1 [4]) and surveys them as the "agentic RAG" design space [5]. It is the
kind of technique researchers reach for right now, and it is invisible inside frameworks,
which is why learners need to see it run.</p>
<p><b>What learners do.</b> The resource is an interactive web application. Learners first
assemble a working naive RAG pipeline over any document they upload (extract &rarr; chunk
&rarr; embed &rarr; retrieve &rarr; generate), deliberately <i>as baseline scaffolding</i>, so
they can feel naive RAG's core failure mode: it trusts whatever the top-k chunks are and answers
once, unverified. Then they flip one toggle and run the same question through the
self-correcting loop, with every decision rendered live: the routing verdict, per-chunk
relevance grades, the draft, the support verdict, query reformulations, and per-call token/latency
costs. A side-by-side compare mode makes the concept's value and its price concrete. All grading
is <i>fail-closed</i>: when a verification step cannot be parsed, the UI reports an unverified,
low-confidence answer, the resource models honest verification rather than hiding failure.</p>

<h2>2&nbsp;&nbsp;Learning Objectives</h2>
<p class="li"><b>LO1.</b> Trace a self-corrective RAG loop, routing, evidence grading, answer
verification, query reformulation, and map each step to its origin in Self-RAG [1] and CRAG [2]
(the trace UI annotates every step with this mapping).</p>
<p class="li"><b>LO2.</b> Explain the accuracy/cost trade-off of reflection using the measured
call counts, tokens, and latencies the tool displays, and steer it with the exposed knobs
(top-k, max correction passes, relevance threshold).</p>
<p class="li"><b>LO3.</b> Identify naive RAG's failure modes, irrelevant retrieval trusted
unconditionally, unsupported claims, reformulation drift, and observe which ones the
self-correcting loop catches, and why verification should fail closed.</p>
<p class="li"><b>LO4.</b> (Scaffolding) Explain each baseline pipeline stage: how chunking
bounds retrievable evidence, how cosine similarity over normalized embeddings ranks it, and why
low-dimensional projections of embedding space are suggestive rather than faithful.</p>
<p class="li"><b>LO5.</b> Contrast plain-LLM, naive-RAG, and self-corrective answers on the
same question and evidence.</p>

<h2>3&nbsp;&nbsp;Prerequisites and Difficulty Levels</h2>
<p><b>Prerequisites.</b> Comfort using a web application; a general idea of what an LLM is.
Vectors/similarity help but are introduced visually. No programming needed to use the tool.</p>
<p><b>Level A (pre-university/intro).</b> Baseline pipeline with defaults; plain-LLM vs RAG
contrast (LO4, LO5). <b>Level B (undergraduate).</b> Vary chunking and embedding models; read
similarity scores and projections critically (LO4). <b>Level C (graduate/practitioner).</b>
The concept itself: agentic mode with the trace open, threshold/loop experiments, cost analysis,
paper mapping (LO1&ndash;LO3). Guided exercises with solutions are included for each level,
grounded in the bundled open-license example corpus (a 1,200-page nutrition textbook).</p>

<h2>4&nbsp;&nbsp;Materials and Accessibility</h2>
<p>The submission contains the full open-source application (MIT; React + FastAPI), setup for
fully local use (Ollama + open embedding models; no paid APIs required), the graded exercise
set with solutions, and the example corpus. A CI-tested suite of 40+ unit/integration tests
accompanies the code. All interactive controls are keyboard-accessible with ARIA roles;
visualizations ship a table-view alternative and captions stating their limits. A transparent
&ldquo;Demo Mode&rdquo; (persistently bannered in the UI and disclosed here) stages one failed
retrieval pass so instructors can reliably demonstrate the correction loop live; all other steps
run genuinely. All materials are original and were created for this track.</p>

<h2>References</h2>
<p class="ref">[1] A. Asai, Z. Wu, Y. Wang, A. Sil, H. Hajishirzi. Self-RAG: Learning to
Retrieve, Generate, and Critique through Self-Reflection. <i>ICLR</i>, 2024. arXiv:2310.11511.</p>
<p class="ref">[2] S.-Q. Yan, J.-C. Gu, Y. Zhu, Z.-H. Ling. Corrective Retrieval Augmented
Generation. arXiv:2401.15884, 2024.</p>
<p class="ref">[3] Z. Jiang et al. Active Retrieval Augmented Generation. <i>EMNLP</i>, 2023.
arXiv:2305.06983.</p>
<p class="ref">[4] B. Jin et al. Search-R1: Training LLMs to Reason and Leverage Search Engines
with Reinforcement Learning. arXiv:2503.09516, 2025.</p>
<p class="ref">[5] A. Singh et al. Agentic Retrieval-Augmented Generation: A Survey on Agentic
RAG. arXiv:2501.09136, 2025.</p>
<p class="ref">[6] P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks. <i>NeurIPS</i>, 2020 (the established baseline this resource uses as scaffolding).</p>
"""

CSS = """
* { font-family: sans-serif; }
h1 { font-size: 14.5px; margin: 0 0 6px 0; }
h2 { font-size: 11.5px; margin: 10px 0 4px 0; }
p  { font-size: 9.2px; line-height: 1.34; margin: 0 0 5px 0; text-align: justify; }
p.meta { font-size: 8.5px; color: #444; margin-bottom: 8px; }
p.li  { margin: 0 0 3px 10px; }
p.ref { font-size: 8.2px; margin: 0 0 2.5px 0; }
"""


def main() -> str:
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag-pipeline-studio-concept.pdf")
    story = fitz.Story(html=HTML, user_css=CSS)
    writer = fitz.DocumentWriter(out_path)
    page_rect = fitz.paper_rect("a4")
    margin = 54
    where = page_rect + (margin, margin, -margin, -margin)
    more = True
    pages = 0
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    print(f"Wrote {out_path} ({pages} pages)")
    return out_path


if __name__ == "__main__":
    main()
