import { matchStepDurations, computeReplayDelays } from '../src/utils/tracePacing.js'
import assert from 'node:assert'

// Two-pass trace: demo-staged grading in pass 1, slow reformulation, real pass 2
const trace = [
  { step: 'route', loop: 0 },
  { step: 'retrieve', loop: 0 },                 // no LLM call
  { step: 'gradeChunks', loop: 0, demo: true },  // staged, no call behind it
  { step: 'reformulate', loop: 0 },              // SLOW call
  { step: 'retrieve', loop: 1 },                 // no LLM call
  { step: 'gradeChunks', loop: 1 },              // real grading call
  { step: 'draftAnswer', loop: 1 },
  { step: 'gradeAnswer', loop: 1 },
]
const callsDetail = [
  { label: 'route', elapsedMs: 180 },
  { label: 'reformulate', elapsedMs: 1900 },  // the slow one
  { label: 'gradeChunks', elapsedMs: 300 },
  { label: 'draftAnswer', elapsedMs: 800 },
  { label: 'gradeAnswer', elapsedMs: 250 },
]

const dur = matchStepDurations(trace, callsDetail)
assert.deepStrictEqual(dur, [180, null, null, 1900, null, 300, 800, 250])
console.log('durations matched:', dur)
// Critical: the demo-staged gradeChunks got null and did NOT steal pass 2's 300ms

const delays = computeReplayDelays(trace, callsDetail)
const slow = delays[3], fast = delays[0]
assert.ok(slow > fast * 5, `reformulate (${slow}ms) must dwell much longer than route (${fast}ms)`)
assert.ok(delays.every(d => d >= 250 && d <= 2500), 'all delays clamped for watchability')
assert.notStrictEqual(new Set(delays).size, 1, 'pacing must NOT be uniform (old 550ms bug)')
console.log('replay delays:', delays)
console.log(`slow reformulation dwells ${slow}ms vs ${fast}ms for routing, rhythm shows real cost ✓`)

// Clamping: a 9s call caps at 2500, a 10ms call floors at 250, ordering preserved
const d2 = computeReplayDelays(
  [{ step: 'a' }, { step: 'b' }],
  [{ label: 'a', elapsedMs: 9000 }, { label: 'b', elapsedMs: 10 }]
)
assert.deepStrictEqual(d2, [2500, 250])
console.log('clamps hold at extremes:', d2)
console.log('\nR2 PASS')
