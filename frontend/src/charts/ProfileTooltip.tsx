import type { Pipe } from '../types'
import { date, format, integer } from '../utils/format'

export function ProfileTooltip({ point }: { point?: Pipe['points'][number] }) {
  if (!point) return null
  return <div className="profile-tooltip">
    <strong>Сегмент #{point.segment_id}</strong>
    <p>{integer(point.position_m)}–{integer(point.end_m)} м</p>
    <dl>
      <div><dt>Прогноз</dt><dd>{format(point.prediction_mm)} мм</dd></div>
      <div><dt>Нижняя граница</dt><dd>{format(point.lower_mm)} мм</dd></div>
      <div><dt>Верхняя граница</dt><dd>{format(point.upper_mm)} мм</dd></div>
      <div><dt>Прошлое ВТД</dt><dd>{point.actual_mm === null ? 'Нет измерения' : `${format(point.actual_mm)} мм`}</dd></div>
    </dl>
    {point.actual_date && <small>Дата ВТД: {date(point.actual_date)}</small>}
    {point.is_risk && <small>Проверить в первую очередь</small>}
  </div>
}
