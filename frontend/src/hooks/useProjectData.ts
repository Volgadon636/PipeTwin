import { useEffect, useState } from 'react'
import { z } from 'zod'
import { importanceSchema, metricsSchema, profileSchema, riskSchema, validationSchema } from '../data/schema'
import type { FrontendData } from '../types'

type State = { status: 'loading' } | { status: 'error'; message: string } | { status: 'ready'; data: FrontendData }

async function read<T>(name: string, schema: z.ZodType<T>, signal: AbortSignal): Promise<T> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}.json`, { signal })
  if (!response.ok) throw new Error(`${name}.json: HTTP ${response.status}`)
  const result = schema.safeParse(await response.json())
  if (!result.success) throw new Error(`${name}.json: данные не соответствуют схеме (${result.error.issues[0]?.path.join('.')})`)
  return result.data
}

export function useProjectData(retry: number): State {
  const [state, setState] = useState<State>({ status: 'loading' })
  useEffect(() => {
    const controller = new AbortController()
    setState({ status: 'loading' })
    void Promise.all([
      read('metrics', metricsSchema, controller.signal),
      read('feature_importance', importanceSchema, controller.signal),
      read('top_risk_segments', riskSchema, controller.signal),
      read('validation', validationSchema, controller.signal),
      read('profile', profileSchema, controller.signal),
    ]).then(([metrics, importance, risk, validation, profile]) => {
      const pipeIds = new Set(profile.pipes.map(pipe => pipe.pipe_id))
      if (risk.rows.some(row => !pipeIds.has(row.pipe_id)) || importance.models.some(model => !pipeIds.has(model.pipe_id)))
        throw new Error('JSON-файлы относятся к разным наборам труб. Повторите экспорт.')
      if (profile.pipes.reduce((sum, pipe) => sum + pipe.points.length, 0) !== metrics.segment_count)
        throw new Error('Количество точек профиля не совпадает с числом сегментов.')
      setState({ status: 'ready', data: { metrics, importance, risk, validation, profile } })
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setState({ status: 'error', message: error instanceof Error ? error.message : 'Не удалось прочитать данные' })
    })
    return () => controller.abort()
  }, [retry])
  return state
}
