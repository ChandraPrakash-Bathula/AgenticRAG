import { useEffect, useState, useMemo } from 'react'
import { computeReplayDelays, matchStepDurations } from '../utils/tracePacing'

const STEP_META = {
  route: { label: 'Route', icon: '🧭' },
  retrieve: { label: 'Retrieve', icon: '🔎' },
  gradeChunks: { label: 'Grade Chunks', icon: '🩺' },
  draftAnswer: { label: 'Generate', icon: '✍️' },
  gradeAnswer: { label: 'Grade Answer', icon: '✅' },
  reformulate: { label: 'Reformulate', icon: '🔁' },
  directAnswer: { label: 'Direct Answer', icon: '💭' },
}

const WHY_TOOLTIP = {
  route: "Self-RAG's \"Retrieve\" reflection token plus FLARE's adaptive retrieval policy, decides up front whether retrieval is even needed before generating anything. Search-R1 later trains this decision rather than prompting it.",
  retrieve: 'The retrieval step shared by both Self-RAG and CRAG, fetches candidate passages from the vector index for the current query.',
  gradeChunks: "CRAG's retrieval evaluator, grades each retrieved passage as relevant/irrelevant before it's allowed into the context (CRAG itself uses a finer Correct/Ambiguous/Incorrect scale).",
  draftAnswer: 'Standard RAG generation, produce an answer conditioned only on the chunks that passed grading.',
  gradeAnswer: "Self-RAG's \"ISSUP\" (is-supported) and \"ISUSE\" (utility, 1 to 5) reflection tokens. ISSUP critiques whether the generation is actually backed by the retrieved evidence, catching unsupported or fabricated claims; ISUSE rates how well it answers the question. They are separate on purpose: a correct \"the document does not say\" is fully supported and nearly useless.",
  reformulate: "CRAG's query-rewriting / knowledge-refinement step, when evidence is insufficient, rewrite the query and retry retrieval instead of answering from weak context.",
  directAnswer: "Self-RAG's no-retrieval branch, the router decided external evidence wasn't necessary, so the model answers directly from its own knowledge.",
}

// One-line collapsed summary, the "headline" of what happened at this step.
function stepSummary(entry) {
  switch (entry.step) {
    case 'route':
      return entry.needsRetrieval ? 'Needs the document → retrieve' : 'Generic query → answer directly'
    case 'retrieve':
      return `"${entry.query}", ${entry.chunks?.length ?? 0} chunks pulled`
    case 'gradeChunks': {
      const relevant = entry.grades?.filter(g => g.relevant).length ?? 0
      return `kept ${relevant} of ${entry.grades?.length ?? 0} as relevant`
    }
    case 'draftAnswer':
      return 'drafted an answer from the kept chunks'
    case 'gradeAnswer':
      return [
        entry.supported ? 'supported ✓' : 'not fully supported ✕',
        `confidence: ${entry.confidence}`,
        entry.utility != null ? `usefulness: ${entry.utility}/5` : null,
      ].filter(Boolean).join(' · ')
    case 'reformulate':
      return `rewrote the query → "${entry.newQuery}"`
    case 'directAnswer':
      return entry.note || 'answered without retrieval'
    default:
      return ''
  }
}

// Join a gradeChunks entry back to the chunk TEXT, which lives on the retrieve
// entry from the same pass, the grader logs only ids + verdicts.
function gradedChunksWithText(entry, trace) {
  const retrieveEntry = trace.find(t => t.step === 'retrieve' && t.loop === entry.loop)
  const byId = {}
  for (const c of retrieveEntry?.chunks || []) byId[String(c.id)] = c
  return (entry.grades || []).map(g => ({
    ...g,
    text: byId[String(g.chunkId)]?.text,
    rank: byId[String(g.chunkId)]?.rank,
  }))
}

// The expanded body: the actual chunks / thinking / decision for this step.
function StepBody({ entry, trace }) {
  switch (entry.step) {
    case 'route':
      return (
        <>
          <span className={`grade-tag ${entry.needsRetrieval ? 'relevant' : 'irrelevant'}`}>
            {entry.needsRetrieval ? 'retrieve' : 'answer directly'}
          </span>
          <p className="trace-detail-text">{entry.reason}</p>
          {entry.note && <div className="trace-node-demo-note">{entry.note}</div>}
        </>
      )

    case 'retrieve':
      return (
        <>
          <div className="trace-detail-label">Search query</div>
          <div className="trace-query">"{entry.query}"</div>
          <div className="trace-detail-label">
            Top {entry.chunks?.length ?? 0} chunks by cosine similarity (before grading)
          </div>
          {(entry.chunks || []).map(c => (
            <div key={c.id} className="trace-chunk">
              <span className="trace-chunk-rank">#{c.rank}</span>
              <span className="trace-chunk-text">{c.text}</span>
            </div>
          ))}
        </>
      )

    case 'gradeChunks': {
      const graded = gradedChunksWithText(entry, trace)
      return (
        <>
          <div className="trace-detail-label">
            The grader read each chunk against the query and kept only the relevant ones
          </div>
          {graded.map(g => (
            <div key={g.chunkId} className={`trace-chunk graded ${g.relevant ? 'relevant' : 'irrelevant'}`}>
              <span className={`grade-tag ${g.relevant ? 'relevant' : 'irrelevant'}`}>
                {g.relevant ? '✓ relevant' : '✕ dropped'}
              </span>
              <span className="trace-chunk-text">{g.text || `chunk ${g.chunkId}`}</span>
            </div>
          ))}
          {entry.note && <div className="trace-node-demo-note">{entry.note}</div>}
        </>
      )
    }

    case 'draftAnswer':
      return (
        <>
          {entry.retrievalQuery && (
            <p className="trace-detail-text muted">
              Chunks were retrieved via the rewritten query, but the answer addresses your original question.
            </p>
          )}
          <div className="trace-detail-label">Draft answer (grounded only in the kept chunks)</div>
          <div className="trace-answer-box">{entry.draftAnswer}</div>
          {entry.note && <div className="trace-node-demo-note">{entry.note}</div>}
        </>
      )

    case 'gradeAnswer':
      return (
        <>
          <div className="trace-detail">
            <span className={`grade-tag ${entry.supported ? 'relevant' : 'irrelevant'}`}>
              {entry.supported ? '✓ supported by sources' : '✕ not fully supported'}
            </span>
            <span className="trace-confidence">confidence: {entry.confidence}</span>
            {entry.utility != null && (
              <span className={`grade-tag ${entry.utility >= 4 ? 'relevant' : entry.utility <= 2 ? 'irrelevant' : ''}`}>
                usefulness {entry.utility}/5
              </span>
            )}
          </div>
          {entry.supported && entry.utility != null && entry.utility <= 2 && (
            <p className="trace-detail-text muted">
              Grounded but not useful: every claim traces back to the chunks, yet it does not
              really answer the question. Support and usefulness are separate judgements.
            </p>
          )}
          {entry.missing ? (
            <div className="trace-critique">
              <strong>Self-critique:</strong> {entry.missing}
            </div>
          ) : (
            <p className="trace-detail-text muted">Every claim traced back to the retrieved chunks.</p>
          )}
        </>
      )

    case 'reformulate':
      return (
        <>
          <p className="trace-detail-text">
            <strong>Why retry:</strong> {entry.reason}
          </p>
          <div className="trace-rewrite">
            <div className="trace-query old">"{entry.query}"</div>
            <div className="trace-rewrite-arrow">rewritten to ↓</div>
            <div className="trace-query new">"{entry.newQuery}"</div>
          </div>
        </>
      )

    case 'directAnswer':
      return (
        <>
          <div className="trace-detail-label">Answered directly, no document lookup needed</div>
          <div className="trace-answer-box">{entry.answer}</div>
          {entry.note && <div className="trace-node-demo-note">{entry.note}</div>}
        </>
      )

    default:
      return null
  }
}

// Parent should remount this component per query (e.g. `key={queryVersion}`)
// so the reveal animation restarts cleanly for each new trace.
export default function AgenticTrace({ trace, loopsUsed, maxLoops, callsDetail }) {
  const [revealCount, setRevealCount] = useState(0)
  // Every step starts expanded, the whole point is that users can SEE what the
  // agent did. `collapsed` holds the stepIds the user has explicitly folded away.
  const [collapsed, setCollapsed] = useState(() => new Set())

  const durations = useMemo(() => matchStepDurations(trace, callsDetail), [trace, callsDetail])

  // Replay pacing follows each step's REAL latency (clamped), not a uniform tick,
  // the rhythm itself shows where the time actually went.
  useEffect(() => {
    if (!trace || trace.length === 0) return
    const delays = computeReplayDelays(trace, callsDetail)
    let cancelled = false
    let timer
    let i = 0
    const revealNext = () => {
      if (cancelled) return
      i += 1
      setRevealCount(i)
      if (i < trace.length) timer = setTimeout(revealNext, delays[i])
    }
    timer = setTimeout(revealNext, delays[0])
    return () => { cancelled = true; clearTimeout(timer) }
  }, [trace, callsDetail])

  if (!trace || trace.length === 0) return null

  const visible = trace.slice(0, revealCount)
  const isAnimating = revealCount < trace.length
  const allCollapsed = visible.length > 0 && visible.every(e => collapsed.has(e.stepId))

  const toggle = (stepId) => {
    setCollapsed(prev => {
      const next = new Set(prev)
      if (next.has(stepId)) next.delete(stepId)
      else next.add(stepId)
      return next
    })
  }

  const toggleAll = () => {
    setCollapsed(allCollapsed ? new Set() : new Set(trace.map(e => e.stepId)))
  }

  return (
    <div className="trace-panel">
      <div className="trace-panel-header">
        <span className="trace-panel-title">🕸️ Agent Trace</span>
        <span
          className="replay-badge"
          title="The pipeline already finished on the server; this replays its decisions. Step pacing is proportional to each call's real latency."
        >
          ▶ replay · paced by real latency
        </span>
        {maxLoops > 0 && (
          <span className="loop-badge">Pass {Math.min(loopsUsed || 1, maxLoops)} of {maxLoops}</span>
        )}
        <button type="button" className="trace-expand-all" onClick={toggleAll}>
          {allCollapsed ? 'Expand all' : 'Collapse all'}
        </button>
      </div>
      <div className="trace-nodes">
        {visible.map((entry, i) => {
          const meta = STEP_META[entry.step] || { label: entry.step, icon: '•' }
          const active = isAnimating && i === visible.length - 1
          const isLoopBack = entry.step === 'reformulate'
          const expanded = !collapsed.has(entry.stepId)
          return (
            <div key={entry.stepId} className="trace-node-wrap">
              {isLoopBack && <div className="trace-loopback">↩ loop back to Retrieve</div>}
              <div className={`trace-node ${active ? 'active' : ''} ${isLoopBack ? 'reformulate' : ''} ${entry.demo ? 'demo' : ''} ${expanded ? 'expanded' : ''}`}>
                <button
                  type="button"
                  className="trace-node-header"
                  onClick={() => toggle(entry.stepId)}
                  aria-expanded={expanded}
                >
                  <span className="trace-node-icon">{meta.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div className="trace-node-label">
                      {meta.label}{entry.loop > 0 ? ` (pass ${entry.loop + 1})` : ''}
                      {entry.demo && <span className="demo-tag">🎬 staged</span>}
                      {WHY_TOOLTIP[entry.step] && (
                        <span
                          className="why-tooltip"
                          data-tooltip={WHY_TOOLTIP[entry.step]}
                          onClick={e => e.stopPropagation()}
                        >ⓘ</span>
                      )}
                    </div>
                    <div className="trace-node-detail">
                      {stepSummary(entry)}
                      {durations[i] != null && <span className="trace-node-ms"> · {Math.round(durations[i])}ms</span>}
                    </div>
                  </div>
                  <span className="trace-node-chevron">▸</span>
                </button>
                {expanded && (
                  <div className="trace-node-body">
                    <StepBody entry={entry} trace={trace} />
                  </div>
                )}
              </div>
              {i < visible.length - 1 && trace[i + 1]?.step !== 'reformulate' && (
                <div className="trace-arrow">↓</div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
