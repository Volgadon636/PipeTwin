import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import type { ValidationGroup } from '../types'
import { AXIS_STYLE, CHART_HEIGHT, COLORS } from '../utils/constants'
import { format } from '../utils/format'

export function ValidationScatter({ group }: { group: ValidationGroup }) {
  const max = Math.ceil(Math.max(...group.rows.flatMap(row => [row.actual_mm, row.prediction_mm])))
  return <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
    <ScatterChart margin={{ top: 12, right: 30, left: 12, bottom: 28 }}>
      <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 4"/>
      <XAxis dataKey="actual_mm" name="Факт" type="number" unit=" мм" domain={[0, max]} tick={AXIS_STYLE} tickLine={false} axisLine={false} label={{ value: 'Измеренная толщина, мм', position: 'bottom', offset: 13, ...AXIS_STYLE }}/>
      <YAxis dataKey="prediction_mm" name="Прогноз" type="number" unit=" мм" domain={[0, max]} tick={AXIS_STYLE} tickLine={false} axisLine={false} label={{ value: 'Прогноз, мм', angle: -90, position: 'insideLeft', ...AXIS_STYLE }}/>
      <ReferenceLine segment={[{ x: 0, y: 0 }, { x: max, y: max }]} stroke="#a5b3be" strokeDasharray="5 5" label={{ value: 'y = x', position: 'insideTopRight', ...AXIS_STYLE }}/>
      <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(value, name) => [`${format(Number(value))} мм`, name]}/>
      <Scatter name={`Валидация ${group.model}`} data={group.rows} fill={COLORS.prediction} fillOpacity={0.55} isAnimationActive={false}/>
    </ScatterChart>
  </ResponsiveContainer>
}
