import { useState, useRef } from 'react'
import axios from 'axios'
import { API } from '../config'
import AgenticTrace from './AgenticTrace'
import ChunkGradeCards from './ChunkGradeCards'
import StepExplainer from './StepExplainer'
import InfoPopover from './InfoPopover'

const SUGGESTED_QUESTIONS = [
  'What is the main topic of this document?',
  'Summarize the key findings.',
  'What methodology was used?',
  'What are the conclusions?',
]

// Qualitative framing for the worked example's 3-way comparison, this is the
// explicit "LLM poor, naive RAG good, agentic RAG very good" story the example
// exists to tell. Wording is a claim about the pattern the curated questions
// are designed to expose, not a guarantee about any single live answer.
const QUALITY_BADGE = {
  plain: { icon: '🧠', label: 'Plain LLM', tag: 'often generic or off, no source', tone: 'poor' },
  naive: { icon: '📚', label: 'Naive RAG', tag: 'grounded in the text, unverified', tone: 'good' },
  agentic: { icon: '🤖', label: 'Agentic RAG', tag: 'grounded + self-verified', tone: 'best' },
}

const CONFIDENCE_LABEL = { high: 'High confidence', medium: 'Medium confidence', low: 'Low confidence' }

// The worked example's curated questions each hinge on ONE narrow chunk of a ~340-page
// excerpt, top_k=3 often pulls 2 topically-adjacent-but-off-target neighbors alongside it,
// so the standard 0.5 relevance ratio (needs 2 of 3 "relevant") fails even when the single
// golden chunk already has the answer. Lower threshold for the example only.
const relevanceThresholdFor = (exampleMode) => (exampleMode ? 0.3 : 0.5)

// LO2: each control is a real accuracy/cost trade-off, not a preference. They are shown
// inline rather than hidden behind an "advanced" disclosure precisely because moving them
// and watching the cost line respond IS the lesson. `effect` names which way the cost goes,
// so the trade-off is legible before you touch the slider.
const TRADEOFF_CONTROLS = [
  {
    key: 'topK',
    label: 'Chunks retrieved (top-k)',
    min: 1, max: 10, step: 1,
    effect: 'More chunks means more evidence and more prompt tokens on every call. Affects both modes.',
  },
  {
    key: 'maxLoops',
    label: 'Max correction passes',
    min: 1, max: 5, step: 1,
    effect: 'More passes means better recovery from weak retrieval, paid for in extra LLM calls and latency. Agentic mode only.',
  },
  {
    key: 'relevanceThreshold',
    label: 'Relevance threshold',
    min: 0, max: 1, step: 0.05,
    effect: 'Fraction of retrieved chunks that must grade relevant before answering. Stricter means more reformulation loops. Agentic mode only.',
  },
]

function formatCost(stats) {
  if (!stats) return null
  const parts = [`${stats.calls} call${stats.calls === 1 ? '' : 's'}`, `${(stats.latencyMs / 1000).toFixed(1)}s`]
  if (stats.totalTokens) parts.push(`${stats.totalTokens} tokens`)
  return parts.join(' · ')
}

function agenticCostNote(agenticResult) {
  const trace = agenticResult?.trace || []
  const caughtUnsupported = trace.some(t => t.step === 'gradeAnswer' && t.supported === false)
  const caughtInsufficientChunks = trace.some(t => {
    if (t.step !== 'gradeChunks' || t.demo) return false
    const total = t.grades?.length || 0
    const relevant = t.grades?.filter(g => g.relevant).length || 0
    return total > 0 && relevant / total < 0.5
  })
  if (caughtUnsupported) return 'caught an unsupported claim and retried'
  if (caughtInsufficientChunks) return 'caught insufficient evidence and reformulated'
  if ((agenticResult?.loopsUsed ?? 0) > 1) return 'needed reformulation to find sufficient evidence'
  return 'passed verification on the first try'
}

export default function StepQuery({ exampleMode, exampleQuestions }) {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState(exampleMode ? 'agentic' : 'naive') // 'naive' | 'agentic'
  const [compareMode, setCompareMode] = useState(!!exampleMode)
  const [demoMode, setDemoMode] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [agenticResult, setAgenticResult] = useState(null)
  const [error, setError] = useState(null)
  const [queryVersion, setQueryVersion] = useState(0)
  const [tuning, setTuning] = useState({
    topK: 3,
    maxLoops: 3,
    relevanceThreshold: relevanceThresholdFor(exampleMode),
  })

  const lastRunStats = agenticResult?.stats || result?.stats || null

  const abortRef = useRef(null)
  const REQUEST_TIMEOUT_MS = 180000 // 3 min, beyond this something is stuck, not slow

  const fetchNaive = (queryText, opts) => axios.post(`${API}/query`, { question: queryText, top_k: tuning.topK }, opts)
  const fetchAgentic = (queryText, opts) => axios.post(`${API}/query-agentic`, {
    question: queryText,
    top_k: tuning.topK,
    max_loops: tuning.maxLoops,
    relevance_threshold: tuning.relevanceThreshold,
    demo_mode: demoMode,
  }, opts)

  const cancelQuery = () => abortRef.current?.abort()

  const runQuery = async (q) => {
    const queryText = q || question
    if (!queryText.trim()) return

    setLoading(true)
    setError(null)
    setQuestion(queryText)
    setResult(null)
    setAgenticResult(null)

    const controller = new AbortController()
    abortRef.current = controller
    const opts = { signal: controller.signal, timeout: REQUEST_TIMEOUT_MS }

    try {
      if (mode === 'naive') {
        const res = await fetchNaive(queryText, opts)
        setResult(res.data)
      } else if (compareMode) {
        const [naiveRes, agenticRes] = await Promise.all([fetchNaive(queryText, opts), fetchAgentic(queryText, opts)])
        setResult(naiveRes.data)
        setAgenticResult(agenticRes.data)
      } else {
        const res = await fetchAgentic(queryText, opts)
        setAgenticResult(res.data)
      }
      setQueryVersion(v => v + 1)
    } catch (err) {
      if (axios.isCancel(err) || err.code === 'ERR_CANCELED') {
        setError('Query cancelled. (Note: the server may still finish the in-flight LLM calls.)')
      } else if (err.code === 'ECONNABORTED') {
        setError('Request timed out after 3 minutes. The backend may be stuck or overloaded. Try again.')
      } else {
        const detail = err.response?.data?.detail || 'Query failed'
        // 400s about missing pipeline state at THIS step mean the server-side session
        // is gone (restart) even though the wizard shows steps completed.
        const stateLoss = err.response?.status === 400 && /Build embeddings first|Select an LLM first/.test(detail)
        setError(stateLoss
          ? `${detail}, the server session was likely reset (server restart). Go back to Step 1 and rerun the pipeline.`
          : detail)
      }
    } finally {
      abortRef.current = null
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <div className="card-header">
        <div className="card-icon">💬</div>
        <div>
          <div className="card-title">Retrieval & QA</div>
          <div className="card-description">Ask questions, compare plain LLM vs RAG answers</div>
        </div>
      </div>

      {exampleMode && (
        <div className="example-mode-banner" role="status" style={{ marginBottom: '16px' }}>
          <span>
            📖 Try the curated questions below. Each targets a specific, textbook-exact figure the
            plain LLM tends to guess wrong or refuse, naive RAG retrieves the real number. Agentic RAG
            re-verifies that same answer, so on an easy question it will often <em>match</em> naive RAG's
            answer rather than "beat" it, the win shows up as a confidence badge and an Agent Trace you
            can audit, not a different number. Its self-checks are themselves live LLM calls, so loop
            count and confidence can vary a little between runs. For a guaranteed self-correction loop
            every time, check 🎬 Demo Mode below.
          </span>
        </div>
      )}

      <StepExplainer
        summary="Ask a question. It gets embedded like the chunks were, the nearest chunks are fetched from the index, and your LLM answers from that evidence. Two modes: Naive answers once and never checks itself; Agentic grades its own evidence and answer, and retries when a check fails."
        details={[
          'Your question → embedding → cosine search in FAISS → the top-k most similar chunks → pasted into the LLM prompt as "context" → answer. That whole path is RAG.',
          'Naive mode also shows a plain-LLM answer (no context) beside the RAG answer, so you can see exactly what retrieval added.',
          'Naive mode\'s weakness: it TRUSTS the top-k chunks unconditionally. If retrieval fetched junk, it still answers from junk, and it can state unsupported things confidently.',
          'Agentic mode adds self-checks around the same pipeline (see the comparison below). Each check is a small extra LLM call, so it costs more and takes longer. Watch the 💵 cost line to see the trade-off.',
        ]}
      >
        <div className="mode-compare">
          <div className={`mode-compare-col ${mode === 'naive' ? 'active-mode' : ''}`}>
            <h4>Naive RAG {mode === 'naive' && '← selected'}</h4>
            <div className="mode-compare-flow">
              question → retrieve top-k<br />
              → answer. done.
            </div>
            ~2 LLM calls (incl. the plain baseline). Fast and cheap. No verification: wrong or irrelevant
            chunks flow straight into the answer, and nothing flags it.
          </div>
          <div className={`mode-compare-col ${mode === 'agentic' ? 'active-mode' : ''}`}>
            <h4>Agentic RAG {mode === 'agentic' && '← selected'}</h4>
            <div className="mode-compare-flow">
              route: “need the doc at all?”<br />
              → retrieve → grade chunks<br />
              → draft → grade answer<br />
              → weak? rewrite query, retry ↺
            </div>
            4–12 LLM calls. Slower and costlier, but it catches irrelevant evidence and unsupported claims,
            retries with a better query, and reports honest confidence, the whole decision path shows up
            in the Agent Trace below.
          </div>
        </div>
      </StepExplainer>

      <div className="mode-toggle" role="group" aria-label="RAG mode">
        <button type="button" className={`mode-toggle-option ${mode === 'naive' ? 'active' : ''}`} aria-pressed={mode === 'naive'} onClick={() => setMode('naive')}>
          Naive RAG
        </button>
        <button type="button" className={`mode-toggle-option ${mode === 'agentic' ? 'active' : ''}`} aria-pressed={mode === 'agentic'} onClick={() => setMode('agentic')}>
          🤖 Agentic RAG
        </button>
      </div>

      {mode === 'agentic' && (
        <div className="agentic-controls">
          <label className="compare-checkbox">
            <input type="checkbox" checked={compareMode} onChange={e => setCompareMode(e.target.checked)} />
            Compare with Naive RAG
          </label>
          <label className="compare-checkbox demo-mode-checkbox">
            <input type="checkbox" checked={demoMode} onChange={e => setDemoMode(e.target.checked)} />
            🎬 Demo Mode (force pass 1 to fail, for reliable live demos)
          </label>
        </div>
      )}

      {exampleMode && exampleQuestions?.length > 0 ? (
        <>
          <div className="section-title">Curated Example Questions</div>
          <div className="example-questions">
            {exampleQuestions.map((eq, i) => (
              <button type="button" key={i} className="example-question-card" disabled={loading} onClick={() => runQuery(eq.question)}>
                <div className="example-question-text">
                  {eq.question}
                  {eq.outOfCorpus && <span className="oob-tag">not in the excerpt</span>}
                </div>
                <div className="example-question-meta">
                  <span className="example-question-chapter">{eq.chapter}</span>
                  <span className="example-question-fact">
                    {/* "Textbook says" is wrong for a question the textbook never answers,
                        and the distinction is the lesson, so the label states which it is. */}
                    {eq.outOfCorpus ? 'What should happen: ' : 'Textbook says: '}{eq.expectedFact}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="section-title">Suggested Questions</div>
          <div className="suggestions">
            {SUGGESTED_QUESTIONS.map((q, i) => (
              <button type="button" key={i} className="suggestion-chip" disabled={loading} onClick={() => runQuery(q)}>{q}</button>
            ))}
          </div>
        </>
      )}

      <div className="query-input-group">
        <input
          className="query-input"
          placeholder="Type your question here..."
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !loading && runQuery()}
        />
        <button className="btn btn-primary" onClick={() => runQuery()} disabled={loading || !question.trim()}>
          {loading ? 'Thinking...' : 'Ask'}
        </button>
      </div>

      {/* LO2: the accuracy/cost trade-off, made movable. The last run's real cost sits
          inside the same panel so a change and its consequence are read together. */}
      <div className="tradeoff-panel">
        <div className="tradeoff-head">
          <h4 className="tradeoff-title">Query-time settings: the accuracy and cost trade-off</h4>
          {lastRunStats
            ? <span className="tradeoff-cost">Last run: {formatCost(lastRunStats)}</span>
            : <span className="tradeoff-cost muted">Run a query to see its cost</span>}
        </div>
        <p className="tradeoff-intro">
          These are <strong>query-time</strong> settings, not a repeat of Steps 2 and 3. Chunking and
          the embedding model rebuild the index; these three change how a question is run against the
          index you already built, so nothing is re-chunked or re-embedded. Every one of them buys
          accuracy with calls, tokens, or latency: move one, ask the same question again, and watch
          the cost line above.
        </p>

        {TRADEOFF_CONTROLS.map(c => {
          const disabled = mode === 'naive' && c.key !== 'topK'
          const value = tuning[c.key]
          return (
            <div className={`slider-group ${disabled ? 'is-disabled' : ''}`} key={c.key}>
              <div className="slider-label">
                <label htmlFor={`tradeoff-${c.key}`}>{c.label}</label>
                <span className="slider-value">
                  {c.step < 1 ? value.toFixed(2) : value}
                </span>
              </div>
              <input
                id={`tradeoff-${c.key}`}
                type="range"
                min={c.min}
                max={c.max}
                step={c.step}
                value={value}
                disabled={disabled}
                aria-describedby={`tradeoff-${c.key}-effect`}
                onChange={e => setTuning(prev => ({ ...prev, [c.key]: +e.target.value }))}
              />
              <div className="slider-hint" id={`tradeoff-${c.key}-effect`}>
                {c.effect}{disabled ? ' Not used in naive mode.' : ''}
              </div>
            </div>
          )
        })}
      </div>

      {loading && (
        <div className="loading-container">
          <div className="spinner" />
          <div className="loading-text">
            {mode === 'agentic' ? 'Running agentic pipeline (route → retrieve → grade → generate → grade)...' : 'Retrieving context and generating answers...'}
          </div>
          <button type="button" className="btn btn-secondary" onClick={cancelQuery} style={{ marginTop: '10px' }}>
            Cancel
          </button>
        </div>
      )}

      {error && <div style={{ color: 'var(--error)', padding: '12px', textAlign: 'center' }}>{error}</div>}

      {mode === 'naive' && result && (
        <div>
          <div className="answers-grid">
            <div className="answer-card">
              <div className="answer-label plain">
                🧠 Plain LLM Answer
                {exampleMode && <span className="quality-tag poor">{QUALITY_BADGE.plain.tag}</span>}
              </div>
              <div className="answer-text">{result.plain_answer}</div>
              {result.plain_truncated && <div className="truncation-note">⚠️ Response hit the model's token limit and may be cut off.</div>}
            </div>
            <div className="answer-card rag">
              <div className="answer-label rag">
                📚 RAG Answer
                {exampleMode && <span className="quality-tag good">{QUALITY_BADGE.naive.tag}</span>}
              </div>
              <div className="answer-text">{result.rag_answer}</div>
              {result.rag_truncated && <div className="truncation-note">⚠️ Response hit the model's token limit and may be cut off.</div>}
              {result.stats && <div className="cost-line">💵 {formatCost(result.stats)}</div>}
            </div>
          </div>

          <div className="sources-section">
            <div className="sources-title">
              📎 Retrieved Sources from {result.source_file}{' '}
              <InfoPopover label="How retrieval picked these">
                Your question was embedded with the same model as the document chunks, and the FAISS
                index returned the top-k chunks whose vectors have the highest cosine similarity to
                the question's vector. Pure meaning-match, no keywords, no exact phrases. Retrieval
                ALWAYS returns k chunks, however weak the best match is; the similarity score is your
                only signal of evidence strength, and in naive mode nothing checks it.
              </InfoPopover>
            </div>
            {result.retrieved_chunks.map((chunk, i) => (
              <div key={i} className="chunk-item">
                <div className="chunk-header">
                  <span className="chunk-index">Rank #{chunk.rank}, Chunk #{chunk.index}</span>
                  <span
                    className="source-score"
                    title="Cosine similarity between this chunk's embedding and the query embedding. Ranges from −1 to 1; higher = more similar. It is NOT a probability, and values are only comparable within a single embedding model."
                  >
                    cosine similarity: {chunk.similarity?.toFixed(3)}
                  </span>
                </div>
                {chunk.text}
              </div>
            ))}
            <div style={{ marginTop: '8px', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
              ℹ️ Similarity is cosine similarity in embedding space (−1 to 1), a ranking signal, not a
              probability or a percentage of relevance.
            </div>
          </div>
        </div>
      )}

      {mode === 'agentic' && agenticResult && (
        <div>
          {demoMode && (
            <div className="demo-mode-banner" role="status">
              🎬 DEMO MODE, pass 1 chunk grading was intentionally forced to fail so the
              self-correction loop always fires. Pass-1 grades below are staged, not real model
              output; every later step ran for real.
            </div>
          )}

          {agenticResult.maxLoopsReached && agenticResult.maxLoopsReason === 'answer_unsupported' && (
            <div className="max-loops-banner severe">
              ⚠️ Self-correction gave up after {agenticResult.loopsUsed} passes, every draft was flagged as unsupported or fabricated. This answer could NOT be verified against the source document. Treat it as low-confidence, not a fact.
            </div>
          )}
          {agenticResult.maxLoopsReached && agenticResult.maxLoopsReason === 'chunk_grading_noise' && (
            <div className="max-loops-banner noise">
              🟡 Retrieval needed {agenticResult.loopsUsed} attempts to find strongly-matching chunks. This answer is grounded in the best content actually found, but relevance grading wasn't fully confident, treat it as tentative, not verified.
            </div>
          )}

          {(() => {
            const showThreeWay = exampleMode && compareMode && !!result
            return (
              <div className={compareMode && result ? `answers-grid ${showThreeWay ? 'answers-grid-3' : ''}` : ''}>
                {showThreeWay && (
                  <div className="answer-card">
                    <div className="answer-label plain">
                      {QUALITY_BADGE.plain.icon} Plain LLM Answer
                      <span className="quality-tag poor">{QUALITY_BADGE.plain.tag}</span>
                    </div>
                    <div className="answer-text">{result.plain_answer}</div>
                    {result.plain_truncated && <div className="truncation-note">⚠️ Response hit the model's token limit and may be cut off.</div>}
                  </div>
                )}
                {compareMode && result && (
                  <div className="answer-card">
                    <div className="answer-label plain">
                      📚 Naive RAG Answer
                      {exampleMode && <span className="quality-tag good">{QUALITY_BADGE.naive.tag}</span>}
                    </div>
                    <div className="answer-text">{result.rag_answer}</div>
                    {result.stats && <div className="cost-line">💵 {formatCost(result.stats)}</div>}
                  </div>
                )}
                <div className={`answer-card rag ${agenticResult.maxLoopsReached ? (agenticResult.maxLoopsReason === 'answer_unsupported' ? 'unverified' : 'tentative') : ''}`}>
                  <div className="answer-label rag">
                    🤖 Agentic RAG Answer
                    {agenticResult.confidence && (
                      <span className={`confidence-badge ${agenticResult.confidence}`}>
                        {CONFIDENCE_LABEL[agenticResult.confidence] || agenticResult.confidence}
                      </span>
                    )}
                    {exampleMode && <span className="quality-tag best">{QUALITY_BADGE.agentic.tag}</span>}
                  </div>
                  <div className="answer-text">{agenticResult.answer}</div>
                  {agenticResult.stats?.truncated && <div className="truncation-note">⚠️ At least one response in this pipeline hit the model's token limit and may be cut off.</div>}
                  {agenticResult.note && <div className="answer-note">{agenticResult.note}</div>}
                  {agenticResult.stats && (
                    <div className="cost-line">💵 {formatCost(agenticResult.stats)}, {agenticCostNote(agenticResult)}</div>
                  )}
                </div>
              </div>
            )
          })()}

          <AgenticTrace
            key={queryVersion}
            trace={agenticResult.trace}
            loopsUsed={agenticResult.loopsUsed}
            maxLoops={agenticResult.maxLoops ?? tuning.maxLoops}
            callsDetail={agenticResult.stats?.calls_detail}
          />

          {agenticResult.retrievedChunks && agenticResult.retrievedChunks.length > 0 && (
            <div className="sources-section">
              <div className="sources-title">
                📎 Graded Sources from {agenticResult.source_file}{' '}
                <InfoPopover label="How these were retrieved & graded">
                  Same cosine top-k retrieval as naive mode, but then a grader LLM (CRAG's
                  "retrieval evaluator") read each chunk against your question and marked it
                  relevant ✓ or irrelevant ✕. Only ✓ chunks were given to the generator; if too few
                  passed the relevance threshold, the query was rewritten and retrieval retried.
                  This grading step is the core difference between naive and agentic RAG.
                </InfoPopover>
              </div>
              <ChunkGradeCards
                chunks={agenticResult.retrievedChunks}
                grades={[...(agenticResult.trace || [])].reverse().find(t => t.step === 'gradeChunks')?.grades}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}
