import assert from 'node:assert/strict'
import test from 'node:test'
import {
  THICKNESS_STOPS,
  clampSegmentIndex,
  getNextRiskIndex,
  getThickness,
  getThicknessColor,
  getThicknessRange,
} from '../src/three/pipeData.ts'

const point = (prediction, actual = null, risk = false) => ({
  prediction_mm: prediction, actual_mm: actual, is_risk: risk,
})
const pipe = points => ({ points })

test('thickness sources keep missing measurements distinct from a real zero', () => {
  assert.equal(getThickness(point(3.2), 'actual'), null)
  assert.equal(getThickness(point(3.2, 0), 'actual'), 0)
  assert.equal(getThickness(point(3.2, 4.1), 'prediction'), 3.2)
  assert.equal(getThickness(point(3.2, 4.1), 'actual'), 4.1)
})

test('one shared range covers both pipes without filling missing measurements', () => {
  const pipes = [pipe([point(7, 6), point(8, null)]), pipe([point(2, 4), point(3, null)])]
  assert.deepEqual(getThicknessRange(pipes, 'prediction'), { min: 2, max: 8 })
  assert.deepEqual(getThicknessRange(pipes, 'actual'), { min: 4, max: 6 })
  assert.equal(getThicknessRange([pipe([point(3), point(4)])], 'actual'), null)
  assert.equal(getThicknessRange([], 'prediction'), null)
})

test('color scale preserves its stops and clamps values outside the displayed range', () => {
  const range = { min: 2, max: 8 }
  assert.equal(getThicknessColor(2, range), THICKNESS_STOPS[0])
  assert.equal(getThicknessColor(5, range), THICKNESS_STOPS[1])
  assert.equal(getThicknessColor(8, range), THICKNESS_STOPS[2])
  assert.equal(getThicknessColor(-1, range), THICKNESS_STOPS[0])
  assert.equal(getThicknessColor(12, range), THICKNESS_STOPS[2])
  assert.equal(getThicknessColor(5, { min: 5, max: 5 }), THICKNESS_STOPS[1])
})

test('missing data and invalid ranges use the missing-data color', () => {
  assert.equal(getThicknessColor(null, { min: 2, max: 8 }), '#8993a0')
  assert.equal(getThicknessColor(3, null), '#8993a0')
  assert.equal(getThicknessColor(NaN, { min: 2, max: 8 }), '#8993a0')
  assert.equal(getThicknessColor(3, { min: 8, max: 2 }), '#8993a0')
})

test('segment indices stay inside the current pipe including the final segment', () => {
  const current = pipe([point(3), point(4), point(5)])
  assert.equal(clampSegmentIndex(current, -4), 0)
  assert.equal(clampSegmentIndex(current, 9), 2)
  assert.equal(clampSegmentIndex(current, 1.6), 2)
  assert.equal(clampSegmentIndex(current, NaN), 0)
  assert.equal(clampSegmentIndex(pipe([]), 2), 0)
})

test('risk navigation selects the strict next or previous array index and never wraps', () => {
  const current = pipe([point(3, null, true), point(4), point(5, null, true), point(6), point(7, null, true)])
  assert.equal(getNextRiskIndex(current, 0, 1), 2)
  assert.equal(getNextRiskIndex(current, 4, -1), 2)
  assert.equal(getNextRiskIndex(current, 1, 1), 2)
  assert.equal(getNextRiskIndex(current, 1, -1), 0)
  assert.equal(getNextRiskIndex(current, 4, 1), null)
  assert.equal(getNextRiskIndex(current, 0, -1), null)
  assert.equal(getNextRiskIndex(current, -1, 1), 0)
  assert.equal(getNextRiskIndex(current, current.points.length, -1), 4)
  assert.equal(getNextRiskIndex(pipe([point(3)]), 0, 1), null)
  assert.equal(getNextRiskIndex(pipe([]), 0, -1), null)
})
