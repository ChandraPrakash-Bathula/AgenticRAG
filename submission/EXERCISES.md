# Guided Exercises, Self-Correcting RAG, Interactively

These exercises use the bundled open-license example corpus: **Human_Nutrition.pdf**
(*Human Nutrition: 2020 Edition*, University of Hawai'i at Mānoa, CC BY-NC-SA, a
1,208-page textbook). Upload it in Step 1 and follow along. Any substantial PDF works,
but the expected observations below are written for this corpus.

Default setup unless an exercise says otherwise: **Recursive chunking (size 500 /
overlap 50) → MiniLM-L6-v2 embeddings → GPT-OSS 20B** (or any local model).

---

## Level A, What does retrieval add? *(no prerequisites)*

### A1. Grounding vs. memory
In Step 5, **Naive mode**, ask:

> *According to the document, what is a kilocalorie?*

**Observe:** both the Plain LLM answer and the RAG answer, and the retrieved source
chunks under them.

**Questions:** (a) Both answers are probably correct, nutrition is common knowledge.
So what did RAG add here, if not correctness? (b) Find the sentence in the retrieved
chunks that the RAG answer is based on.

<details><summary>Solution</summary>

(a) **Verifiability, not just knowledge.** The plain LLM answers from memory, you
cannot check where its claim came from. The RAG answer is grounded in visible chunks.
Grounding matters most when the model's memory is wrong or the document is private/new;
here it demonstrates the *audit trail*. (b) The definition sentence appears in the
top-ranked chunk; note its cosine similarity score. **Bonus observation:** the book
defines "kilocalorie" in two ways in different chapters, thermodynamically (the heat
needed to raise 1 kg of water by 1°C) and as 1,000 calories (the label "Calorie").
Retrieval typically surfaces the thermodynamic one, the chunk whose *embedding* is
closest, not necessarily the definition you expected. Ask *"How many calories are in a
kilocalorie according to the document?"* and watch retrieval fetch the other passage:
the phrasing of your question steers which evidence is found. That sensitivity is a
core RAG behavior worth feeling early.
</details>

### A2. When the document can't answer
Same setup. Ask:

> *Who won the 2024 Olympic marathon?*

**Observe:** the RAG answer and the similarity scores on the retrieved chunks.

<details><summary>Solution</summary>

The RAG answer should say the context doesn't contain this (the prompt instructs the
model to answer ONLY from context). The chunks are still shown, retrieval **always**
returns the top-k nearest chunks, however weak, but their cosine similarities are
noticeably lower than in A1. Lesson: retrieval has no built-in "no good match" signal;
naive RAG relies entirely on the generator noticing. Remember this for Level C.
</details>

---

## Level B, The knobs that shape retrieval *(basic vectors helpful)*

### B1. Chunk size bounds what can be retrieved
Run the pipeline twice: **Fixed chunking at 200 chars**, then re-run Step 2 with
**Fixed at 2000 chars** (note: re-running resets later steps, that's intentional).
Both times, ask:

> *What are the functions of fiber in the body?*

**Compare:** number of chunks, the retrieved chunks' completeness, and the answers.

<details><summary>Solution</summary>

At 200 chars the fiber discussion (around page 48: fiber is a complex carbohydrate that
digestive enzymes cannot break down) is shredded across fragments, retrieval finds
*a* relevant fragment but the model sees clipped context. At 2000 chars each chunk
carries whole paragraphs, fuller answers, but each retrieved chunk also carries more
off-topic text and more prompt tokens. Neither is "right": chunk size is a
precision-vs-context trade-off, and it is set **before** any question is asked.
</details>

### B2. The embedding model changes what "similar" means
Keep chunking fixed. Run Step 3 with **MiniLM**, ask A1's question, note the top-3
chunk indexes and similarities. Re-run Step 3 with **MPNet-Base-v2** and repeat.

<details><summary>Solution</summary>

The ranking usually overlaps but rarely matches exactly, and the similarity *values*
differ systematically. Two lessons: (1) "similarity" is model-defined, not absolute,
each model embeds meaning into its own geometry; (2) scores are **not comparable across
models** (the UI caption says exactly this). This is why swapping embedding models can
silently change RAG behavior in production systems.
</details>

### B3. Don't trust the map
In Step 3, open the **2D view**, pick two points that look like close neighbors, then
check them in the **Table view** and the **3D view**.

<details><summary>Solution</summary>

Often the "neighbors" are less close in 3D, or their texts are only loosely related.
UMAP/t-SNE squash hundreds of dimensions into 2–3 and distort distances doing it,
the caption under the plot states this. Retrieval runs in the *full* space; the plot is
for exploration, never for measurement.
</details>

---

## Level C, The concept: self-correcting retrieval *(the emerging technique)*

### C1. Read a healthy trace
Switch Step 5 to **Agentic mode**, check **Compare with Naive RAG**, and ask:

> *According to the document, how is vitamin D related to bone health?*

**Observe:** the Agent Trace (hover each step's ⓘ), the graded source cards, the
confidence badge, and the 💵 cost lines of both answers.

<details><summary>Solution</summary>

Map the steps first: *route* is Self-RAG's retrieve-decision, *gradeChunks* is CRAG's
retrieval evaluator, *gradeAnswer* is Self-RAG's ISSUP (is-supported) critique,
*reformulate* is CRAG's corrective query rewrite. On a small focused document this is
often a single clean pass; on this 1,200-page book expect **one or two real
reformulations** before chunk grading accepts the evidence, vitamin D is discussed in
many scattered places, and early retrievals mix in weak matches. That's not a bug; it
IS the concept: the loop notices weak evidence and re-queries, and it still ends in a
supported, high-confidence answer. Cost: check the 💵 line, typically 4–9 calls vs
naive's 1 for the same question. **Reflection is not free**; that asymmetry is the
concept's core trade-off.
</details>

### C2. Force the correction loop
First, ask something genuinely vague with default settings:

> *Is it healthy?*

**Observe the route step.** Then, in the **Query-time settings: the accuracy and cost trade-off**
panel below the question box, set **Relevance threshold to 1.00** (every retrieved chunk
must be graded relevant), and ask a vague question that is *anchored to the document*:

> *According to this document, is it healthy?*

**Observe:** reformulate steps in the trace, the loop-back arrows, the final banner and
confidence, and the total cost.

<details><summary>Solution</summary>

Part 1 is a trick: the **router** classifies bare *"Is it healthy?"* as a generic
question needing no document and answers directly, no retrieval, no loops. That's
Self-RAG's retrieve-decision doing its job (and the app's deterministic safety net
never fires, because nothing in the query references the document). Part 2 anchors the
question to the document, so retrieval runs, and with a maximally strict threshold,
chunk grading keeps failing on such a vague query: the query is rewritten (each rewrite
starts from your ORIGINAL question, to prevent drift) and retrieval retries, often to
max passes and an honestly-labeled low-confidence answer. Drop the threshold to 0.3
and re-ask: fewer or no loops. The threshold is a strictness dial: higher = more
skepticism, more retries, more cost. Note the final answer still addresses your
original question, not the rewrite.
</details>

### C3. Watch it catch a failure (staged, disclosed)
Enable **🎬 Demo Mode** and ask A1's kilocalorie question again.

<details><summary>Solution</summary>

Pass 1's chunk grading is *staged* to fail (banner + "🎬 staged" tags say so, nothing
is hidden), forcing a real reformulation and a real second pass that recovers. This is
the loop's value proposition compressed into one run: a system that notices weak
evidence and retries beats one that answers anyway. Compare with A2, where naive mode
had no such recourse.
</details>

### C4. The ill-posed question *(compare mode)*
In **Agentic mode + Compare with Naive RAG**, ask the suggested question:

> *What are the conclusions?*

This question is written for a *paper*, a 1,200-page textbook has no conclusions
section. **Observe:** what each mode does with an unanswerable-as-posed question.

<details><summary>Solution</summary>

Naive RAG answers **confidently anyway**: it fetches the 3 nearest chunks, often
Chapter 1's iodized-salt case study, whose text is full of conclusion-flavored
language, and presents one anecdote as "the conclusions," with no signal that
anything is wrong. Agentic mode typically struggles honestly: chunk grading and/or
answer grading reject passes, reformulation retries, and the result arrives labeled
**low confidence** with a max-loops banner (or, on some runs, a borderline pass,
grading is an LLM judgment and vague questions sit on the boundary; the variance is
itself informative). The contrast is the whole lesson: **naive fails silently,
agentic fails loudly.** Neither mode can fix an ill-posed question, only the
question-asker can.
</details>

### C5. Why fail closed? *(discussion)*
The pipeline's graders sometimes return unparseable output. This tool then treats
chunks as **not relevant** and answers as **unsupported / low confidence**, the
strictest reading. The lazier default (assume relevant/supported) would make the UI
look *most* confident exactly when its verifier is broken.

**Question:** connect this to verification systems generally, tests, type checkers,
safety checks. When is fail-open ever acceptable?

<details><summary>Solution (sketch)</summary>

A verifier that fails open converts its own outages into false assurance, the
consumer cannot distinguish "verified" from "verifier was down." Fail-open is only
tolerable when the check is advisory and its absence is visible. In this tool the
check's *output is the product* (the confidence badge), so it must fail closed, and
the trace additionally discloses the grader failure. Self-RAG/CRAG papers assume the
critic works; deploying reflection means engineering for when it doesn't.
</details>
