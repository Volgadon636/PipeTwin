import type { z } from 'zod'
import type { importanceSchema, metricsSchema, profileSchema, riskRowSchema, riskSchema, validationSchema } from '../data/schema'

export type MetricsData = z.infer<typeof metricsSchema>
export type ImportanceData = z.infer<typeof importanceSchema>
export type RiskData = z.infer<typeof riskSchema>
export type RiskRow = z.infer<typeof riskRowSchema>
export type ProfileData = z.infer<typeof profileSchema>
export type Pipe = ProfileData['pipes'][number]
export type ValidationData = z.infer<typeof validationSchema>
export type ValidationGroup = ValidationData['groups'][number]
export interface FrontendData {
  metrics: MetricsData
  importance: ImportanceData
  risk: RiskData
  validation: ValidationData
  profile: ProfileData
}
export interface SegmentSelection { pipeId: string; segmentId: number }
