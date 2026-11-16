import type { Pipe } from '../types'
import type { ThicknessRange, ThicknessSource } from './pipeData'

export type PipeViewMode = 'outside' | 'inside'
export interface PipeSceneView {
  index: number
  mode: PipeViewMode
  source: ThicknessSource
  range: ThicknessRange | null
  resetToken: number
}
export interface PipeSceneProps extends PipeSceneView {
  pipe: Pipe
  onIndexChange: (index: number) => void
}
export interface PipeSceneController {
  update: (view: PipeSceneView) => void
  dispose: () => void
}
