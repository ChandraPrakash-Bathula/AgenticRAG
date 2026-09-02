/**
 * Paper vs implementation diagrams for Self-RAG and CRAG.
 *
 * Drawing only the published mechanism would imply this playground implements all of
 * it, which it does not: there are no trained reflection tokens, no utility scoring,
 * no ternary retrieval verdict, and no web-search fallback. So each figure is a pair,
 * published on the left and implemented on the right, with the gaps named and costed.
 * Approximating a research method under real constraints is itself the lesson.
 *
 * Every claim here was checked against backend/app/agentic_rag.py rather than the
 * papers' abstracts.
 */

const W = 300
const BOX = { w: 244, h: 34, x: 28 }

function Box({ y, label, tone = '', sub }) {
  return (
    <g>
      <rect className={`mech-box ${tone}`} x={BOX.x} y={y} width={BOX.w} height={BOX.h} rx="6" />
      <text className="mech-label" x={W / 2} y={y + (sub ? 15 : 22)} textAnchor="middle">{label}</text>
      {sub && <text className="mech-sub" x={W / 2} y={y + 27} textAnchor="middle">{sub}</text>}
    </g>
  )
}

function Arrow({ from, to, label }) {
  const mid = (from + to) / 2
  return (
    <g>
      <line className="mech-arrow" x1={W / 2} y1={from} x2={W / 2} y2={to - 6} markerEnd="url(#mech-tip)" />
      {label && <text className="mech-edge" x={W / 2 + 6} y={mid + 3}>{label}</text>}
    </g>
  )
}

function Defs() {
  return (
    <defs>
      <marker id="mech-tip" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">
        <path className="mech-tip" d="M0,0 L8,4 L0,8 z" />
      </marker>
    </defs>
  )
}

/* Self-RAG as published: one trained model emits reflection tokens inline while
   generating, and candidate continuations are ranked by those token scores. */
function SelfRagPaper() {
  return (
    <svg viewBox={`0 0 ${W} 424`} role="img" aria-label="Self-RAG as published" className="mech-svg">
      <Defs />
      <Box y={8} label="query" tone="mech-io" />
      <Arrow from={42} to={62} />
      <Box y={62} label="Retrieve token" sub="retrieve / no-retrieve / continue" tone="mech-decide" />
      <Arrow from={96} to={116} />
      <Box y={116} label="retrieve K passages" />
      <Arrow from={150} to={170} />
      <Box y={170} label="generate a candidate per passage" sub="in parallel" />
      <Arrow from={204} to={224} />
      <Box y={224} label="ISREL" sub="is the passage relevant" tone="mech-score" />
      <Arrow from={258} to={278} />
      <Box y={278} label="ISSUP" sub="is the claim supported" tone="mech-score" />
      <Arrow from={312} to={332} />
      <Box y={332} label="ISUSE" sub="how useful, 1 to 5" tone="mech-score" />
      <Arrow from={366} to={386} />
      <Box y={386} label="beam search over token scores" tone="mech-io" />
    </svg>
  )
}

/* What this playground runs: the same four decision points, but each is a separate
   prompted call to an off-the-shelf model, and there is one draft rather than a beam. */
function SelfRagApp() {
  return (
    <svg viewBox={`0 0 ${W} 424`} role="img" aria-label="Self-RAG in this playground" className="mech-svg">
      <Defs />
      <Box y={8} label="query" tone="mech-io" />
      <Arrow from={42} to={62} />
      <Box y={62} label="route" sub="needsRetrieval true / false" tone="mech-decide" />
      <Arrow from={96} to={116} label="true" />
      <Box y={116} label="retrieve top-k" sub="FAISS cosine" />
      <Arrow from={150} to={170} />
      <Box y={170} label="gradeChunks" sub="relevant true / false per chunk" tone="mech-score" />
      <Arrow from={204} to={224} />
      <Box y={224} label="draft one answer" sub="from kept chunks only" />
      <Arrow from={258} to={278} />
      <Box y={278} label="gradeAnswer.supported" sub="plus what is missing" tone="mech-score" />
      <Arrow from={312} to={332} />
      <Box y={332} label="gradeAnswer.utility" sub="how useful, 1 to 5" tone="mech-score" />
      <Arrow from={366} to={386} />
      <Box y={386} label="answer + confidence + utility" tone="mech-io" />
    </svg>
  )
}

/* CRAG as published: a lightweight evaluator returns a ternary verdict, and each
   branch repairs the evidence differently before generation. */
function CragPaper() {
  return (
    <svg viewBox={`0 0 ${W} 400`} role="img" aria-label="CRAG as published" className="mech-svg">
      <Defs />
      <Box y={8} label="query" tone="mech-io" />
      <Arrow from={42} to={62} />
      <Box y={62} label="retrieve" />
      <Arrow from={96} to={116} />
      <Box y={116} label="retrieval evaluator" sub="correct / ambiguous / incorrect" tone="mech-decide" />
      <Arrow from={150} to={178} />
      <g>
        <rect className="mech-box mech-branch" x={10} y={178} width={90} height={44} rx="6" />
        <text className="mech-label sm" x={55} y={196} textAnchor="middle">correct</text>
        <text className="mech-sub" x={55} y={210} textAnchor="middle">refine only</text>
        <rect className="mech-box mech-branch" x={106} y={178} width={90} height={44} rx="6" />
        <text className="mech-label sm" x={151} y={196} textAnchor="middle">ambiguous</text>
        <text className="mech-sub" x={151} y={210} textAnchor="middle">refine + web</text>
        <rect className="mech-box mech-branch" x={202} y={178} width={90} height={44} rx="6" />
        <text className="mech-label sm" x={247} y={196} textAnchor="middle">incorrect</text>
        <text className="mech-sub" x={247} y={210} textAnchor="middle">web search</text>
      </g>
      <Arrow from={222} to={250} />
      <Box y={250} label="knowledge refinement" sub="decompose, filter, recompose" />
      <Arrow from={284} to={304} />
      <Box y={304} label="generate" />
      <Arrow from={338} to={358} />
      <Box y={358} label="answer" tone="mech-io" />
    </svg>
  )
}

/* The playground keeps the corrective loop but collapses the verdict to a ratio test
   and repairs the QUERY rather than the evidence, since there is no web index. */
function CragApp() {
  return (
    <svg viewBox={`0 0 ${W} 400`} role="img" aria-label="CRAG in this playground" className="mech-svg">
      <Defs />
      <Box y={8} label="query" tone="mech-io" />
      <Arrow from={42} to={62} />
      <Box y={62} label="retrieve top-k" />
      <Arrow from={96} to={116} />
      <Box y={116} label="gradeChunks" sub="relevant fraction vs threshold" tone="mech-decide" />
      <Arrow from={150} to={178} />
      <g>
        <rect className="mech-box mech-branch" x={22} y={178} width={124} height={44} rx="6" />
        <text className="mech-label sm" x={84} y={196} textAnchor="middle">above threshold</text>
        <text className="mech-sub" x={84} y={210} textAnchor="middle">draft the answer</text>
        <rect className="mech-box mech-loop" x={154} y={178} width={124} height={44} rx="6" />
        <text className="mech-label sm" x={216} y={196} textAnchor="middle">below threshold</text>
        <text className="mech-sub" x={216} y={210} textAnchor="middle">rewrite the query</text>
      </g>
      <Arrow from={222} to={250} />
      <Box y={250} label="gradeAnswer" sub="supported true / false" tone="mech-score" />
      <Arrow from={284} to={304} label="unsupported" />
      <Box y={304} label="reformulate and retry" sub="capped at max passes" tone="mech-loop" />
      <Arrow from={338} to={358} />
      <Box y={358} label="answer + confidence" tone="mech-io" />
    </svg>
  )
}

const SELF_RAG_DELTAS = [
  ['Trained reflection tokens', 'Separate prompted calls to an off-the-shelf model',
   'Self-RAG fine-tunes a model to emit critique tokens while generating. Reproducing that needs training. Prompting a stock model for the same four judgements keeps the decision points visible without a training run.'],
  ['Beam search over candidates', 'One draft, then verify',
   'The beam is what makes Self-RAG strong and also what hides the reasoning. One draft plus one explicit check is legible in a trace panel.'],
]

const CRAG_DELTAS = [
  ['Correct / ambiguous / incorrect', 'Binary relevant per chunk, then a ratio test',
   'A ternary verdict needs a calibrated evaluator. A per-chunk boolean plus an adjustable threshold gets a learner to the same decision, and the threshold is a slider they can move.'],
  ['Web-search fallback', 'Not implemented',
   'CRAG reaches outside the corpus when retrieval fails. This playground stays inside your uploaded document on purpose: the lesson is what your own document can and cannot answer.'],
  ['Knowledge refinement of retrieved text', 'Reformulation of the query instead',
   'CRAG repairs the evidence by decomposing and recomposing strips. The playground repairs the query and retrieves again, which is the same corrective intent with one moving part.'],
]

function Deltas({ rows }) {
  return (
    <div className="mech-deltas">
      {rows.map(([paper, app, why]) => (
        <div className="mech-delta" key={paper}>
          <div className="mech-delta-head">
            <span className="mech-delta-paper">{paper}</span>
            <span className="mech-delta-arrow" aria-hidden="true">&rarr;</span>
            <span className="mech-delta-app">{app}</span>
          </div>
          <p className="mech-delta-why">{why}</p>
        </div>
      ))}
    </div>
  )
}

function Figure({ id, n, title, cite, href, lead, Paper, App, deltas, caption }) {
  return (
    <section className="mech-figure" id={id} aria-labelledby={`${id}-h`}>
      <h3 className="home-section-title" id={`${id}-h`}>
        Figure {n}. {title}
      </h3>
      <p className="home-section-intro">
        {lead}{' '}
        <a className="mech-cite" href={href} target="_blank" rel="noreferrer">{cite}</a>
      </p>

      <div className="mech-pair">
        <div className="mech-col">
          <div className="mech-col-head">As published</div>
          <Paper />
        </div>
        <div className="mech-col">
          <div className="mech-col-head">In this playground</div>
          <App />
        </div>
      </div>
      <p className="mech-caption">{caption}</p>

      <h4 className="mech-deltas-title">What differs, and why</h4>
      <Deltas rows={deltas} />
    </section>
  )
}

export default function Mechanisms({ onStart }) {
  return (
    <div className="home mech">
      <section className="home-hero">
        <p className="home-eyebrow">Visual explanation</p>
        <h2 className="home-headline">Two papers, and what this playground actually runs.</h2>
        <p className="home-lede">
          The agentic mode is an approximation of Self-RAG and CRAG, not a reimplementation.
          Showing only the published mechanisms would suggest otherwise, so each figure below puts
          the paper beside the running code and names every gap. Reading the two together is the
          point: it teaches the method and the cost of approximating it without a training run,
          a web index, or a calibrated evaluator.
        </p>
      </section>

      <Figure
        id="self-rag" n={1}
        title="Self-RAG, reflection as generation"
        lead="Self-RAG trains a single model to interleave critique tokens with its own output, deciding when to retrieve and scoring what it produced."
        cite="Asai et al., 2023, arXiv:2310.11511"
        href="https://arxiv.org/abs/2310.11511"
        Paper={SelfRagPaper} App={SelfRagApp}
        deltas={SELF_RAG_DELTAS}
        caption="All three reflection signals are here: relevance, support, and utility. What differs is order and means. Self-RAG generates first and scores the candidates it produced; the playground filters the evidence first, drafts once from what survived, then checks that draft on both axes. And the judge differs: a fine-tuned model emitting tokens mid-generation, versus separate prompted calls whose inputs and outputs you can read in the trace."
      />

      <Figure
        id="crag" n={2}
        title="CRAG, correction after retrieval"
        lead="CRAG scores retrieval quality with a lightweight evaluator and repairs the evidence before generating, reaching outside the corpus when it has to."
        cite="Yan et al., 2024, arXiv:2401.15884"
        href="https://arxiv.org/abs/2401.15884"
        Paper={CragPaper} App={CragApp}
        deltas={CRAG_DELTAS}
        caption="The corrective loop survives; the repair target changes. CRAG fixes the retrieved text, the playground fixes the query and retrieves again, capped so a failing question terminates instead of looping forever."
      />

      <section className="home-section" aria-labelledby="beyond-h">
        <h3 className="home-section-title" id="beyond-h">One thing neither paper specifies</h3>
        <p className="home-section-intro">
          Both papers assume their critique step returns a usable verdict. In practice a grading
          call can return malformed output, and a pipeline that treats an unreadable verdict as a
          pass will report fabricated content as verified.
        </p>
        <p className="home-note">
          Every grading step here fails closed. An unparseable chunk grade counts as irrelevant, an
          unparseable support verdict counts as unsupported with low confidence, and the trace says
          which happened. That is an engineering decision this resource adds, not something either
          paper prescribes, and it is why the answer panel can say <strong>unverified</strong>
          {' '}rather than quietly saying nothing.
        </p>
      </section>

      <section className="home-cta">
        <p>See these run on a real question.</p>
        <button type="button" className="btn btn-primary" onClick={onStart}>
          Open the playground
        </button>
      </section>
    </div>
  )
}
