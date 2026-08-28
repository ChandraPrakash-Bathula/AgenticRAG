/**
 * Landing page: what the playground teaches, topic by topic, before the wizard starts.
 *
 * Content is deliberately duplicated in spirit (not in wording) with each step's
 * StepExplainer. The explainers answer "what am I looking at right now"; this page
 * answers "why does this step exist at all", which is the question you have before
 * you have seen the step.
 */

const TOPICS = [
  {
    n: 1,
    icon: '📄',
    title: 'Document ingestion',
    learn: 'How a PDF becomes plain text, and why that extraction step is the first place a pipeline can quietly lose information.',
    why: 'A language model cannot read a PDF. It reads tokens. Everything downstream inherits whatever the text layer gave you, so a scanned page with no text layer produces a pipeline that retrieves nothing and answers confidently anyway.',
    app: 'Upload any PDF, or load a worked example built from an open nutrition textbook.',
  },
  {
    n: 2,
    icon: '✂️',
    title: 'Chunking',
    learn: 'Four strategies for splitting text (fixed size, recursive character, sentence based, semantic paragraph) and what each one does to the boundaries of an idea.',
    why: 'A chunk is the smallest unit of evidence your system can ever retrieve. Split a definition in half and no amount of clever searching will find it whole. Chunk too large and you bury one useful sentence inside noise the model must wade through.',
    app: 'Switch strategies and watch the same document break apart differently.',
  },
  {
    n: 3,
    icon: '🧬',
    title: 'Embeddings and vector search',
    learn: 'How text becomes a vector, why similar meanings land near each other, and how nearest neighbour search over those vectors turns into retrieval.',
    why: 'This is what makes RAG semantic rather than keyword based. A question about "daily protein needs" can match a passage that never uses the word "daily", because the match happens in meaning space rather than string space.',
    app: 'Five embedding models, with the resulting vector space projected to 2D and 3D so you can see the clusters.',
  },
  {
    n: 4,
    icon: '🤖',
    title: 'Model selection',
    learn: 'The difference between the model that finds evidence and the model that writes the answer, and how much answer quality actually depends on model size.',
    why: 'These are two separate jobs and people routinely conflate them. Retrieval quality is set by your embedding model and your chunks. Generation quality is set by the model reading them. Pick the smallest model here and you will see how much of the work retrieval was doing all along.',
    app: 'Cloud models from 7B to 120B via Groq, or local models through Ollama.',
  },
  {
    n: 5,
    icon: '💬',
    title: 'Retrieval and generation',
    learn: 'The full loop: question in, embedding, nearest chunks, prompt assembly, answer out. Then the same question through a self correcting pipeline, side by side.',
    why: 'Seeing a plain model answer, a naive RAG answer, and a self verified answer to one question is the fastest way to understand what retrieval adds and what it still fails to guarantee.',
    app: 'Ask anything. Compare all three answers with the full agent trace.',
  },
]

// Split deliberately: CORE are the papers this resource actually teaches and cites in
// the accompanying write-up, so the app and the paper must not drift apart. SUPPORTING
// are the pipeline internals a learner may want to read next.
const CORE_REFERENCES = [
  {
    authors: 'Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.',
    year: '2023',
    title: 'Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection',
    venue: 'ICLR 2024',
    id: 'arXiv:2310.11511',
    href: 'https://arxiv.org/abs/2310.11511',
    note: 'Originates the reflection-token mechanism. The route and gradeAnswer steps in the trace are modelled on it.',
  },
  {
    authors: 'Yan, S., Gu, J., Zhu, Y., Ling, Z.',
    year: '2024',
    title: 'Corrective Retrieval Augmented Generation',
    venue: 'Preprint',
    id: 'arXiv:2401.15884',
    href: 'https://arxiv.org/abs/2401.15884',
    note: 'Introduces the retrieval evaluator and corrective rewriting that the gradeChunks and reformulate steps implement.',
  },
  {
    authors: 'Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., Neubig, G.',
    year: '2023',
    title: 'Active Retrieval Augmented Generation (FLARE)',
    venue: 'EMNLP 2023',
    id: 'arXiv:2305.06983',
    href: 'https://arxiv.org/abs/2305.06983',
    note: 'Contributes the adaptive retrieval policy underlying the routing decision: retrieve only when the model needs it.',
  },
  {
    authors: 'Jin, B., Zeng, H., Yue, Z., Wang, D., Zamani, H., Han, J.',
    year: '2025',
    title: 'Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning',
    venue: 'Preprint',
    id: 'arXiv:2503.09516',
    href: 'https://arxiv.org/abs/2503.09516',
    note: 'Trains a model to decide on its own when to search, operationalizing the routing decision this trace panel makes visible.',
  },
  {
    authors: 'Singh, A., Ehtesham, A., Kumar, S., Khoei, T.T.',
    year: '2025',
    title: 'Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG',
    venue: 'Preprint',
    id: 'arXiv:2501.09136',
    href: 'https://arxiv.org/abs/2501.09136',
    note: 'Treats this reflect, verify, retry loop as its own active design space in current agentic systems.',
  },
  {
    authors: 'Lewis, P., Perez, E., Piktus, A., et al.',
    year: '2020',
    title: 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    venue: 'NeurIPS 2020',
    id: 'arXiv:2005.11401',
    href: 'https://arxiv.org/abs/2005.11401',
    note: 'The paper that named RAG. Steps 1 through 5 are this architecture, before any self-correction is added.',
  },
]

const SUPPORTING_REFERENCES = [
  {
    authors: 'Karpukhin, V., Oğuz, B., Min, S., et al.',
    year: '2020',
    title: 'Dense Passage Retrieval for Open-Domain Question Answering',
    venue: 'EMNLP 2020',
    id: 'arXiv:2004.04906',
    href: 'https://arxiv.org/abs/2004.04906',
    note: 'Why dense vector retrieval beats keyword search for question answering.',
  },
  {
    authors: 'Reimers, N., Gurevych, I.',
    year: '2019',
    title: 'Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks',
    venue: 'EMNLP 2019',
    id: 'arXiv:1908.10084',
    href: 'https://arxiv.org/abs/1908.10084',
    note: 'The sentence embedding approach behind the models in Step 3.',
  },
  {
    authors: 'Johnson, J., Douze, M., Jégou, H.',
    year: '2019',
    title: 'Billion-scale Similarity Search with GPUs',
    venue: 'IEEE Transactions on Big Data',
    id: 'arXiv:1702.08734',
    href: 'https://arxiv.org/abs/1702.08734',
    note: 'FAISS, the index this playground searches at query time.',
  },
  {
    authors: 'McInnes, L., Healy, J., Melville, J.',
    year: '2018',
    title: 'UMAP: Uniform Manifold Approximation and Projection',
    venue: 'Preprint',
    id: 'arXiv:1802.03426',
    href: 'https://arxiv.org/abs/1802.03426',
    note: 'How the embedding plots in Step 3 get from many dimensions down to two.',
  },
  {
    authors: 'Gao, Y., Xiong, Y., Gao, X., et al.',
    year: '2023',
    title: 'Retrieval-Augmented Generation for Large Language Models: A Survey',
    venue: 'Preprint',
    id: 'arXiv:2312.10997',
    href: 'https://arxiv.org/abs/2312.10997',
    note: 'A map of the wider field if you want to read further.',
  },
]

function Reference({ r }) {
  return (
    <li className="home-ref">
      <span className="home-ref-authors">{r.authors}</span>{' '}
      <span className="home-ref-year">({r.year}).</span>{' '}
      <a className="home-ref-title" href={r.href} target="_blank" rel="noreferrer">{r.title}</a>.{' '}
      <span className="home-ref-venue">{r.venue}.</span>{' '}
      <span className="home-ref-id">{r.id}</span>
      <span className="home-ref-note">{r.note}</span>
    </li>
  )
}

export default function Home({ onStart }) {
  return (
    <div className="home">
      <section className="home-hero">
        <p className="home-eyebrow">Interactive playground</p>
        <h2 className="home-headline">
          Build a RAG pipeline one step at a time, then watch it correct itself.
        </h2>
        <p className="home-lede">
          Retrieval augmented generation is usually taught as a diagram. This playground makes you
          assemble the thing instead: upload a document, split it, embed it, pick a model, and ask a
          question. Once the naive pipeline works, one toggle runs the same question through a self
          correcting loop so you can see exactly what the extra machinery buys and what it costs.
        </p>
        <div className="home-actions">
          <button type="button" className="btn btn-primary" onClick={onStart}>
            Open the playground
          </button>
          <a className="home-jump" href="#references">Jump to references</a>
        </div>
      </section>

      <section className="home-section" aria-labelledby="pipeline-heading">
        <h3 className="home-section-title" id="pipeline-heading">What you will learn</h3>
        <p className="home-section-intro">
          Five steps, in the order the data actually flows. Each one is a place real pipelines break.
        </p>

        <ol className="home-topics">
          {TOPICS.map(t => (
            <li className="home-topic" key={t.n}>
              <div className="home-topic-head">
                <span className="home-topic-n">{t.n}</span>
                <h4 className="home-topic-title">
                  <span aria-hidden="true" className="home-topic-icon">{t.icon}</span>
                  {t.title}
                </h4>
              </div>
              <div className="home-topic-grid">
                <div className="home-topic-cell">
                  <div className="home-topic-label">What you will learn</div>
                  <p className="home-topic-text">{t.learn}</p>
                </div>
                <div className="home-topic-cell">
                  <div className="home-topic-label">Why it matters</div>
                  <p className="home-topic-text">{t.why}</p>
                </div>
              </div>
              <p className="home-topic-app"><strong>In the app:</strong> {t.app}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="home-section" aria-labelledby="lesson-heading">
        <h3 className="home-section-title" id="lesson-heading">The main lesson</h3>
        <p className="home-section-intro">
          Everything above builds the baseline. This is the comparison the playground exists for.
        </p>

        <div className="home-compare">
          <div className="home-compare-col">
            <h4>Naive RAG</h4>
            <pre className="home-flow">{`question
  ↓ embed
  ↓ retrieve top-k
  ↓ answer
done`}</pre>
            <p>
              Roughly two model calls. Fast and cheap, and it trusts its retrieved chunks
              unconditionally. If the search returned the wrong passages, this pipeline answers from
              the wrong passages with exactly the same confidence, and nothing anywhere flags it.
            </p>
          </div>
          <div className="home-compare-col">
            <h4>Agentic RAG</h4>
            <pre className="home-flow">{`question
  ↓ route: does this need the document?
  ├─ no  → answer directly, stop here
  └─ yes ↓
       retrieve top-k
       ↓ grade every chunk
       ↺ too few relevant? rewrite query, retry
       ↓ draft answer
       ↓ grade the answer against those chunks
       ↺ unsupported? rewrite query, retry
     answer + confidence`}</pre>
            <p>
              Two to twelve model calls. Slower and costlier, and it catches irrelevant evidence and
              unsupported claims, rewrites its own query, and reports honest confidence. Every
              decision is recorded in a trace you can audit step by step.
            </p>
          </div>
        </div>

        <p className="home-note">
          The interesting result is not that one always wins. It is that you can watch the cost of
          verification in calls and milliseconds, and decide for yourself when that cost is worth
          paying.
        </p>

        <div className="home-callout">
          <h4>Why your trace may be shorter than the diagram</h4>
          <p>
            Both retry arrows are conditional, so a healthy run skips them. Three things shorten a trace,
            and all three are the pipeline working correctly:
          </p>
          <ul>
            <li>
              <strong>The router answered directly.</strong> Ask something your document cannot help with
              and routing stops there, with nothing retrieved and therefore nothing to grade. Ask something
              your document actually covers to see the rest.
            </li>
            <li>
              <strong>Retrieval succeeded first time.</strong> If every chunk grades relevant and the draft
              answer is fully supported, there is nothing to correct and the run ends in one pass.
            </li>
            <li>
              <strong>You are in naive mode.</strong> Naive RAG has no grading stage at all. That is the
              point of the comparison.
            </li>
          </ul>
          <p>
            To watch the full correction loop on demand, tick <strong>Demo Mode</strong> in the Query step.
            It forces the first chunk grading pass to fail so reformulation and retry always fire. Staged
            steps are labelled in the trace, and every pass after the first is real.
          </p>
        </div>
      </section>

      <section className="home-section" aria-labelledby="paths-heading">
        <h3 className="home-section-title" id="paths-heading">Two ways to start</h3>
        <div className="home-paths">
          <div className="home-path">
            <h4>Bring your own PDF</h4>
            <p>
              Best for seeing how your own material behaves. Chunking and retrieval quality depend
              heavily on how a document is written, and that is worth feeling directly.
            </p>
          </div>
          <div className="home-path">
            <h4>Load the worked example</h4>
            <p>
              Fast forwards the first three steps with a pre chunked excerpt of an open nutrition
              textbook, and supplies curated questions whose answers are known exact figures. Best
              for a first pass, or for teaching in front of an audience.
            </p>
          </div>
        </div>
      </section>

      <section className="home-section" aria-labelledby="openness-heading">
        <h3 className="home-section-title" id="openness-heading">What runs where, and what is open</h3>
        <p className="home-section-intro">
          Nothing in this pipeline depends on a closed model. Here is the honest split, because
          "runs in the cloud" and "is a proprietary model" are two different things.
        </p>

        <div className="home-paths">
          <div className="home-path">
            <h4>Always on your machine</h4>
            <p>
              Text extraction, chunking, embeddings, and the vector index. The five embedding models
              are open sentence-transformers pulled from HuggingFace and run in process, so the
              document you upload and the vectors built from it never leave the machine. The entire
              retrieval half of RAG is local regardless of which generator you pick.
            </p>
          </div>
          <div className="home-path">
            <h4>Your choice for generation</h4>
            <p>
              Only the generation and grading calls need a model server. Point them at Ollama and the
              whole pipeline is offline. Point them at Groq and they run on open-weight models that
              Groq hosts: Groq is an inference provider, not a model vendor, and every model offered
              here is downloadable from HuggingFace. The default provider is Ollama; Groq is opt in.
            </p>
          </div>
        </div>

        <p className="home-note">
          The cloud path exists for one reason: a 120B model answering in under two seconds makes the
          accuracy and cost comparison legible in a classroom. The same comparison runs on a laptop,
          just slower.
        </p>
      </section>

      <section className="home-section" id="references" aria-labelledby="references-heading">
        <h3 className="home-section-title" id="references-heading">References</h3>
        <p className="home-section-intro">
          The papers this playground is built on. Every mechanism in the app traces back to one of these,
          and the agent trace names the relevant one on each step as it runs.
        </p>

        <h4 className="home-refs-group">Core: the mechanisms this resource teaches</h4>
        <ol className="home-refs">
          {CORE_REFERENCES.map(r => <Reference key={r.id} r={r} />)}
        </ol>

        <h4 className="home-refs-group">Supporting: the pipeline internals</h4>
        <ol className="home-refs" start={CORE_REFERENCES.length + 1}>
          {SUPPORTING_REFERENCES.map(r => <Reference key={r.id} r={r} />)}
        </ol>
      </section>

      <section className="home-cta">
        <p>Ready to build it?</p>
        <button type="button" className="btn btn-primary" onClick={onStart}>
          Open the playground
        </button>
      </section>
    </div>
  )
}
