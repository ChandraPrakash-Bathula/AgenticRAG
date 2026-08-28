// Pure helpers for pacing the agent-trace replay with REAL per-call latencies.
// The trace is a replay of a pipeline that already finished server-side; revealing
// steps at a uniform tick would misrepresent where the time went. Instead each
// step's reveal delay is its actual elapsedMs (clamped for watchability), so a slow
// reformulation visibly takes longer than a fast routing call.

const DEFAULTS = {
  minDelay: 250,    // even instant steps get a beat, or the replay is unreadable
  maxDelay: 2500,   // cap so one slow call doesn't stall the replay
  noCallDelay: 400, // steps with no LLM call behind them (retrieve, staged demo grades)
}

/**
 * Match each trace entry to the duration of the LLM call behind it, or null if the
 * step made no call. Both lists are chronological; each entry consumes the first
 * remaining call whose label equals the step name. Demo-staged entries never match,
 * they had no real call, and letting them consume one would steal a later pass's
 * duration.
 */
export function matchStepDurations(trace, callsDetail) {
  const remaining = [...(callsDetail || [])]
  return (trace || []).map(entry => {
    if (entry.demo) return null
    const idx = remaining.findIndex(c => c.label === entry.step)
    if (idx === -1) return null
    const [call] = remaining.splice(idx, 1)
    return call.elapsedMs ?? null
  })
}

/**
 * Reveal delay per step: the step's real duration, clamped to [minDelay, maxDelay];
 * steps without an LLM call get a short fixed beat.
 */
export function computeReplayDelays(trace, callsDetail, opts = {}) {
  const { minDelay, maxDelay, noCallDelay } = { ...DEFAULTS, ...opts }
  return matchStepDurations(trace, callsDetail).map(ms =>
    ms == null ? noCallDelay : Math.max(minDelay, Math.min(maxDelay, ms))
  )
}
