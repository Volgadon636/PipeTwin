import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'
import { metricsSchema, importanceSchema, riskSchema, validationSchema, profileSchema } from '../src/data/schema.ts'

const read = name => JSON.parse(readFileSync(new URL(`../public/data/${name}.json`, import.meta.url), 'utf8'))

test('all five documents satisfy the current schema and share a release', () => {
  const documents = [
    metricsSchema.parse(read('metrics')), importanceSchema.parse(read('feature_importance')),
    riskSchema.parse(read('top_risk_segments')), validationSchema.parse(read('validation')),
    profileSchema.parse(read('profile')),
  ]
  assert.equal(new Set(documents.map(document => document.release_id)).size, 1)
  assert.deepEqual(documents[0].headlines.map(row => [row.case, row.model]), [['S1', 'H18'], ['S2', 'H31']])
})

test('the former v1 interval contract cannot silently load as the current release', () => {
  assert.equal(profileSchema.safeParse({ ...read('profile'), schema_version: 1 }).success, false)
  const invalid = structuredClone(read('profile'))
  invalid.pipes[0].points[0].lower_mm = invalid.pipes[0].points[0].prediction_mm + 1
  assert.equal(profileSchema.safeParse(invalid).success, false)
})

test('all target segments carry finite intervals and correct final endpoints', () => {
  const profile = profileSchema.parse(read('profile'))
  assert.deepEqual(profile.pipes.map(pipe => [pipe.segment_count, pipe.risk_count]), [[525, 53], [1475, 148]])
  assert.deepEqual(profile.pipes.map(pipe => pipe.points.at(-1).end_m), [5248.4, 14745.1])
  for (const pipe of profile.pipes) {
    for (const point of pipe.points) {
      assert.ok(point.lower_mm <= point.prediction_mm && point.prediction_mm <= point.upper_mm)
    }
  }
})

test('negative and undefined profile correlations remain visible alongside measured coverage', () => {
  const groups = validationSchema.parse(read('validation')).groups
  const retro = groups.find(group => group.id === 'S2_retro_2023')
  assert.equal(retro.profiles.filter(row => row.metrics.pearson_r < 0).length, 2)
  assert.ok(retro.profiles.some(row => row.metrics.pearson_squared === null))
  assert.ok(groups.find(group => group.id === 'S1_intervals').metrics.coverage < .8)
  assert.ok(groups.find(group => group.id === 'S2_intervals').metrics.coverage > .8)
})
