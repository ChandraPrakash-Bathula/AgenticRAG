// Pure helpers for wizard step-invalidation logic (kept framework-free so they can
// be unit-tested directly with Node).

// Which pipelineData key belongs to each step. Step 5 (Query) holds no pipeline data.
export const STEP_DATA_KEYS = {
  1: 'pdfData',
  2: 'chunkData',
  3: 'embedData',
  4: 'llmModel',
}

/**
 * Completing (or RE-completing) a step invalidates everything downstream of it,
 * the backend already resets its downstream state (new chunks wipe embeddings etc.),
 * so the UI must not keep later steps checkmarked. Returns the new completed set.
 */
export function completeAndInvalidate(completedSteps, step) {
  const next = new Set([...completedSteps].filter(s => s <= step))
  next.add(step)
  return next
}

/**
 * A step's action re-ran successfully (without the user clicking Continue yet):
 * keep that step's completion mark if it had one, but drop everything after it.
 */
export function invalidateDownstream(completedSteps, step) {
  return new Set([...completedSteps].filter(s => s <= step))
}

/**
 * Merge a step's fresh data into pipelineData while clearing all downstream keys.
 */
export function mergeAndClearDownstream(pipelineData, step, data) {
  const next = { ...pipelineData, ...data }
  for (const [s, key] of Object.entries(STEP_DATA_KEYS)) {
    if (Number(s) > step) next[key] = null
  }
  return next
}
