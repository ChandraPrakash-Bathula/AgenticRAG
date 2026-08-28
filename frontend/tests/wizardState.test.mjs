import { completeAndInvalidate, invalidateDownstream, mergeAndClearDownstream } from '../src/utils/wizardState.js'
import assert from 'node:assert'

// Full wizard completed once
let completed = new Set()
let data = {}
for (const [step, d] of [[1,{pdfData:'P'}],[2,{chunkData:'C'}],[3,{embedData:'E'}],[4,{llmModel:'L'}]]) {
  completed = completeAndInvalidate(completed, step)
  data = mergeAndClearDownstream(data, step, d)
}
completed = completeAndInvalidate(completed, 5)
assert.deepStrictEqual([...completed].sort(), [1,2,3,4,5])
assert.deepStrictEqual(data, { pdfData:'P', chunkData:'C', embedData:'E', llmModel:'L' })
console.log('full run: steps 1-5 complete, all data present ✓')

// User goes back and RE-RUNS chunking (action succeeds, no Continue click yet)
completed = invalidateDownstream(completed, 2)
data = mergeAndClearDownstream(data, 2, {})
assert.deepStrictEqual([...completed].sort(), [1,2], 'steps 3-5 must un-checkmark')
assert.strictEqual(data.embedData, null)
assert.strictEqual(data.llmModel, null)
assert.strictEqual(data.pdfData, 'P', 'upstream data must survive')
console.log('re-run chunking: steps 3-5 dropped, embed/llm data cleared, pdf kept ✓')

// The wizard's own gate (goToStep logic) now refuses a jump to Query
const canGoTo = (stepId, currentStep) => completed.has(stepId) || stepId === currentStep
assert.strictEqual(canGoTo(5, 2), false, 'jump to Query must be gated')
assert.strictEqual(canGoTo(1, 2), true)
console.log('gating: jump to Query blocked, back-nav to step 1 allowed ✓')

// Re-completing step 2 via Continue also invalidates (covers the other path)
completed = completeAndInvalidate(new Set([1,2,3,4,5]), 2)
assert.deepStrictEqual([...completed].sort(), [1,2])
console.log('completeStep path invalidates downstream too ✓')
console.log('\nR4 PASS')
