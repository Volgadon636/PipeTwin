import { useEffect, useState } from 'react'
import { SectionHeading } from '../components/SectionHeading'
import { Tabs } from '../components/Tabs'
import { ThicknessProfile } from '../charts/ThicknessProfile'
import type { ProfileData, SegmentSelection } from '../types'
import { date, format, integer } from '../utils/format'

export function ProfileSection({ data, intervalMethod, selection }: { data: ProfileData; intervalMethod: string; selection: SegmentSelection | null }) {
  const [pipeId, setPipeId] = useState(data.pipes[0].pipe_id)
  useEffect(() => { if (selection) setPipeId(selection.pipeId) }, [selection])
  const pipe = data.pipes.find(item => item.pipe_id === pipeId) ?? data.pipes[0]
  const selectedSegment = selection?.pipeId === pipeId ? selection.segmentId : undefined
  const selected = pipe.points.find(point => point.segment_id === selectedSegment)
  return <section id="profile" className="panel">
    <SectionHeading title="Профиль толщины вдоль трубы"
      description={`Прогноз на ${date(pipe.prediction_date)} · ${pipe.case === 'cold_start' ? 'холодный старт' : 'между ВТД'} · ${format(pipe.length_m / 1000)} км`}
      aside={<Tabs label="Выбрать трубу" options={data.pipes.map(item => ({ id: item.pipe_id, label: item.pipe_id }))} value={pipeId} onChange={setPipeId}/>}/>
    <div className="chart-card profile-card"><div className="profile-toolbar"><span>Сегментов: {integer(pipe.segment_count)} · в приоритете: {integer(pipe.risk_count)}</span><div className="legend"><span className="legend-line prediction">Прогноз</span>{pipe.case !== 'cold_start' && <span className="legend-line actual">Прошлое ВТД</span>}<span className="legend-band">P10–P90</span><span className="legend-risk">Проверить</span></div></div>
      <ThicknessProfile key={pipeId} pipe={pipe} selectedSegment={selectedSegment}/>
      {selected && <div className="selected-segment"><strong>Сегмент #{selected.segment_id}</strong><span>Позиция: {integer(selected.position_m)} м</span><span>Прогноз: {format(selected.prediction_mm)} мм</span><span>P10: {format(selected.p10_mm)} мм</span><span>P90: {format(selected.p90_mm)} мм</span></div>}
      <p className="footnote">{pipe.case === 'cold_start' ? 'Для холодного старта измерений ВТД/УЗТ нет.' : 'Факт — прошлое ВТД, выполненное до даты прогноза.'} Нижняя полоса — выбор участка для приближения.</p>
    </div>
    <div className="interval-note"><strong>Интервал P10–P90</strong><span>Сценарные границы · покрытие на валидации не сохранено</span></div>
    <details className="method-details"><summary>Как получены границы</summary><p>{intervalMethod}</p></details>
  </section>
}
