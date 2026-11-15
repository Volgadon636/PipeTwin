import { Area, Brush, CartesianGrid, ComposedChart, Line, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Pipe } from '../types'
import { AXIS_STYLE, COLORS, PROFILE_HEIGHT } from '../utils/constants'
import { integer } from '../utils/format'
import { ProfileTooltip } from './ProfileTooltip'

export function ThicknessProfile({ pipe, selectedSegment }: { pipe: Pipe; selectedSegment?: number }) {
  const points = pipe.points.map(point => ({ ...point, interval: [point.p10_mm, point.p90_mm] }))
  const selected = pipe.points.find(point => point.segment_id === selectedSegment)
  return <ResponsiveContainer width="100%" height={PROFILE_HEIGHT}>
    <ComposedChart data={points} margin={{ top: 20, right: 32, left: 5, bottom: 18 }}>
      <CartesianGrid stroke={COLORS.grid} vertical={false}/>
      <XAxis dataKey="position_m" type="number" domain={[0, pipe.length_m]} height={52} tick={AXIS_STYLE} tickFormatter={integer} tickLine={false} axisLine={false} label={{ value: 'Позиция вдоль трубы, м', position: 'insideBottom', offset: 8, ...AXIS_STYLE }}/>
      <YAxis tick={AXIS_STYLE} unit=" мм" tickLine={false} axisLine={false} width={64} domain={[0, 'auto']}/>
      <Tooltip content={({ active, payload }) => {
        if (!active || !payload?.length) return null
        const position = Number(payload[0].payload.position_m)
        return <ProfileTooltip point={pipe.points.find(point => point.position_m === position)}/>
      }}/>
      <Area dataKey="interval" name="Сценарные границы P10–P90" type="linear" fill={COLORS.prediction} fillOpacity={0.12} stroke="none" isAnimationActive={false}/>
      <Line dataKey="actual_mm" name="Прошлое ВТД" type="linear" stroke={COLORS.actual} strokeWidth={1.3} dot={false} connectNulls={false} isAnimationActive={false}/>
      <Line dataKey="prediction_mm" name="Прогноз v2" type="linear" stroke={COLORS.prediction} strokeWidth={1.7} dot={false} isAnimationActive={false}/>
      {pipe.points.filter(point => point.is_risk).map(point => <ReferenceDot key={point.segment_id} x={point.position_m} y={point.prediction_mm} r={2.4} fill={COLORS.risk} stroke="none"/>) }
      {selected && <ReferenceDot x={selected.position_m} y={selected.prediction_mm} r={6} fill={COLORS.risk} stroke="white" strokeWidth={2} label={{ value: `#${selected.segment_id}`, position: 'top', fill: COLORS.risk, fontSize: 12 }}/>}
      <Brush dataKey="position_m" height={26} stroke="#b6c8df" fill="#f7f9fc" tickFormatter={value => `${integer(Number(value))} м`} travellerWidth={8}/>
    </ComposedChart>
  </ResponsiveContainer>
}
