import type { Pipe } from '../types'

export type ThicknessSource = 'prediction' | 'actual'
export type ThicknessRange = { min: number; max: number }

export const THICKNESS_STOPS = ['#c86b49', '#d5b976', '#65a9aa'] as const

const MISSING_COLOR = '#8993a0'

export function getThickness(point: Pipe['points'][number], source: ThicknessSource): number | null {
  const value = source === 'prediction' ? point.prediction_mm : point.actual_mm
  return value !== null && Number.isFinite(value) ? value : null
}

export function getThicknessRange(pipes: readonly Pipe[], source: ThicknessSource): ThicknessRange | null {
  let min = Infinity
  let max = -Infinity
  for (const pipe of pipes) {
    for (const point of pipe.points) {
      const value = getThickness(point, source)
      if (value === null) continue
      min = Math.min(min, value)
      max = Math.max(max, value)
    }
  }
  return min === Infinity ? null : { min, max }
}

export function getThicknessColor(value: number | null, range: ThicknessRange | null): string {
  if (value === null || !Number.isFinite(value) || range === null ||
      !Number.isFinite(range.min) || !Number.isFinite(range.max) || range.max < range.min) {
    return MISSING_COLOR
  }
  if (range.max === range.min) return THICKNESS_STOPS[1]
  const normalized = Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)))
  const scaled = normalized * (THICKNESS_STOPS.length - 1)
  const index = Math.min(Math.floor(scaled), THICKNESS_STOPS.length - 2)
  const start = THICKNESS_STOPS[index]
  const end = THICKNESS_STOPS[index + 1]
  const weight = scaled - index
  const channels = [1, 3, 5].map(offset => {
    const first = parseInt(start.slice(offset, offset + 2), 16)
    const second = parseInt(end.slice(offset, offset + 2), 16)
    return Math.round(first + (second - first) * weight).toString(16).padStart(2, '0')
  })
  return `#${channels.join('')}`
}

export function clampSegmentIndex(pipe: Pipe, index: number): number {
  if (pipe.points.length === 0 || Number.isNaN(index)) return 0
  return Math.max(0, Math.min(Math.round(index), pipe.points.length - 1))
}

export function getNextRiskIndex(pipe: Pipe, current: number, direction: 1 | -1): number | null {
  if (Number.isNaN(current)) return null
  const start = direction === 1
    ? Math.max(0, Math.floor(current) + 1)
    : Math.min(pipe.points.length - 1, Math.ceil(current) - 1)
  for (let index = start; index >= 0 && index < pipe.points.length; index += direction) {
    if (pipe.points[index].is_risk) return index
  }
  return null
}
