import { z } from 'zod'

const number = z.number().finite()
const nullable = number.nullable()
export const metricSchema = z.object({
  n: number.int().nonnegative(), mae_mm: nullable, rmse_mm: nullable,
  bias_mm: nullable, r2: nullable, pearson_r: nullable, pearson_squared: nullable,
  coverage: nullable, mean_interval_width_mm: nullable,
})
const version = { schema_version: z.literal(1) }
export const metricsSchema = z.object({
  ...version, final_model: z.string(), model_status: z.string(),
  segment_count: number.int().positive(), total_length_m: number.positive(), segment_size_m: number.positive(),
  headline: metricSchema, comparison_note: z.string(),
  comparison: z.array(z.object({
    id: z.string(), label: z.string(), cohort: z.string(), final: z.boolean(), metrics: metricSchema,
  })),
})
export const importanceSchema = z.object({
  ...version, method: z.string().nullable(), description: z.string(), available: z.boolean(),
  known_factors: z.array(z.string()), source: z.string(), physical_model: z.string(), models: z.array(z.object({
    pipe_id: z.string(), case: z.string(), features: z.array(z.object({
      feature: z.string(), label: z.string(), value: number.nonnegative(),
    })),
  })),
})
export const riskRowSchema = z.object({
  segment_id: number.int().nonnegative(), pipe_id: z.string(), position_m: number.nonnegative(),
  prediction_mm: number.nonnegative(), p10_mm: number.nonnegative(), p90_mm: number.nonnegative(),
  rank: number.int().positive(), risk_level: z.string(), criterion: z.string(),
}).refine(row => row.p10_mm <= row.p90_mm, 'Invalid risk interval')
export const riskSchema = z.object({
  ...version, fraction: number.positive().max(1), criterion: z.string(), interval_note: z.string(), rows: z.array(riskRowSchema),
})
export const validationSchema = z.object({
  ...version, nominal_coverage: number.min(0).max(1).nullable(), interval_method: z.string(),
  groups: z.array(z.object({
    id: z.string(), label: z.string(), model: z.string(), description: z.string(), metrics: metricSchema,
    rows: z.array(z.object({
      pipe_id: z.string(), segment_id: number.int(), position_m: number, date: z.string(),
      actual_mm: number, prediction_mm: number, p10_mm: nullable, p90_mm: nullable,
    })),
  })).nonempty(),
})
export const profileSchema = z.object({
  ...version, actual_note: z.string(), pipes: z.array(z.object({
    pipe_id: z.string(), label: z.string(), case: z.enum(['cold_start', 'between_vtd']),
    prediction_date: z.string(), length_m: number.positive(), segment_count: number.int(),
    risk_count: number.int(), baseline_missing_count: number.int(),
    points: z.array(z.object({
      segment_id: number.int(), position_m: number, end_m: number,
      actual_mm: nullable, actual_date: z.string().nullable(), prediction_mm: number,
      p10_mm: number, p90_mm: number, is_risk: z.boolean(),
    }).refine(row => row.p10_mm <= row.p90_mm, 'Invalid profile interval')),
  })).min(2),
})
