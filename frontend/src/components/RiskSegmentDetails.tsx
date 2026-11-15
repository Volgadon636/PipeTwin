import type { Pipe, RiskRow, SegmentSelection } from '../types'
import { date, format, integer } from '../utils/format'

export function RiskSegmentDetails({ row, pipe, onSelect }: {
  row: RiskRow
  pipe?: Pipe
  onSelect: (selection: SegmentSelection) => void
}) {
  const point = pipe?.points.find(item => item.segment_id === row.segment_id)
  return <div className="segment-details">
    <div className="segment-details-heading">
      <div>
        <strong>Сегмент #{row.segment_id} · труба {row.pipe_id}</strong>
        <p>Место в списке для этой трубы: {row.rank}{pipe && ` из ${pipe.risk_count}`}</p>
      </div>
      <button className="button" onClick={() => onSelect({ pipeId: row.pipe_id, segmentId: row.segment_id })}>
        Показать на профиле ↑
      </button>
    </div>
    <dl className="segment-details-grid">
      <div><dt>Участок трубы</dt><dd>{integer(row.position_m)}{point && `–${integer(point.end_m)}`} м</dd></div>
      <div><dt>Прогноз толщины</dt><dd>{format(row.prediction_mm)} мм</dd></div>
      <div><dt>Границы P10–P90</dt><dd>{format(row.p10_mm)}–{format(row.p90_mm)} мм</dd></div>
      <div><dt>Дата прогноза</dt><dd>{pipe ? date(pipe.prediction_date) : '—'}</dd></div>
      <div><dt>Прошлое ВТД</dt><dd>{point?.actual_mm == null ? 'Нет измерения' : `${format(point.actual_mm)} мм`}
        {point?.actual_date && <small>от {date(point.actual_date)}</small>}
      </dd></div>
      <div><dt>Сценарий</dt><dd>{pipe ? pipe.case === 'cold_start' ? 'Холодный старт' : 'Между ВТД' : '—'}</dd></div>
    </dl>
    <p className="segment-details-criterion">Критерий приоритета: {row.criterion.toLocaleLowerCase('ru-RU')}.</p>
  </div>
}
