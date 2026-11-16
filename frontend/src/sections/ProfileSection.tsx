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
  const intervalLevel = format(pipe.interval_summary.nominal_coverage * 100, 0)
  return <section id="profile" className="panel">
    <SectionHeading title="Профиль толщины вдоль трубы"
      description={`${pipe.model} · прогноз на ${date(pipe.prediction_date)} · ${pipe.case === 'cold_start' ? 'холодный старт' : 'между ВТД'} · ${format(pipe.length_m / 1000)} км`}
      aside={<Tabs label="Выбрать трубу" options={data.pipes.map(item => ({ id: item.pipe_id, label: item.pipe_id }))} value={pipeId} onChange={setPipeId}/>}/>
    <div className="chart-card profile-card"><div className="profile-toolbar"><span>Сегментов: {integer(pipe.segment_count)} · в приоритете: {integer(pipe.risk_count)}</span><div className="legend"><span className="legend-line prediction">Прогноз</span>{pipe.case !== 'cold_start' && <span className="legend-line actual">Прошлое ВТД</span>}<span className="legend-band">{intervalLevel}%-интервал</span><span className="legend-risk">Проверить</span></div></div>
      <ThicknessProfile key={pipeId} pipe={pipe} selectedSegment={selectedSegment}/>
      {selected && <div className="selected-segment"><strong>Сегмент #{selected.segment_id}</strong><span>Позиция: {integer(selected.position_m)} м</span><span>Прогноз: {format(selected.prediction_mm)} мм</span><span>Нижняя: {format(selected.lower_mm)} мм</span><span>Верхняя: {format(selected.upper_mm)} мм</span></div>}
      <p className="footnote">{pipe.case === 'cold_start' ? 'Для холодного старта собственные ВТД/УЗТ не используются.' : `Линия факта — прямые замеры ВТД 06.07.2022; ${pipe.baseline_missing_count} пропусков не заполнены.`} Нижняя полоса — выбор участка для приближения.</p>
    </div>
    <div className="interval-note"><strong>Эмпирический {intervalLevel}%-интервал</strong><span>Покрытие на исключённых участках: {format(pipe.interval_summary.coverage * 100, 1)} % · {pipe.interval_summary.n} наблюдений{pipe.interval_summary.coverage < pipe.interval_summary.nominal_coverage ? ` · уровень ${intervalLevel} % не подтверждён` : ''}</span></div>
    <details className="method-details"><summary>Как получены границы и что проверено</summary><p>{intervalMethod}</p></details>
  </section>
}
