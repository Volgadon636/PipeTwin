import { useMemo, useState } from 'react'
import { SectionHeading } from '../components/SectionHeading'
import type { RiskData, RiskRow, SegmentSelection } from '../types'
import { PAGE_SIZE } from '../utils/constants'
import { format, integer } from '../utils/format'

type SortKey = 'pipe_id' | 'segment_id' | 'position_m' | 'prediction_mm' | 'p10_mm' | 'p90_mm' | 'risk_level'
const columns: { key: SortKey; label: string }[] = [
  { key: 'pipe_id', label: 'Труба' }, { key: 'segment_id', label: 'Сегмент' },
  { key: 'position_m', label: 'Позиция, м' }, { key: 'prediction_mm', label: 'Прогноз, мм' },
  { key: 'p10_mm', label: 'P10, мм' }, { key: 'p90_mm', label: 'P90, мм' },
  { key: 'risk_level', label: 'Проверка' },
]

export function RiskSection({ data, onSelect }: { data: RiskData; onSelect: (selection: SegmentSelection) => void }) {
  const [pipe, setPipe] = useState('all')
  const [sortKey, setSortKey] = useState<SortKey>('prediction_mm')
  const [ascending, setAscending] = useState(true)
  const [page, setPage] = useState(0)
  const pipeIds = [...new Set(data.rows.map(row => row.pipe_id))]
  const rows = useMemo(() => data.rows.filter(row => pipe === 'all' || row.pipe_id === pipe).sort((a, b) => {
    const first = a[sortKey], second = b[sortKey]
    const order = typeof first === 'number' && typeof second === 'number' ? first - second : String(first).localeCompare(String(second), 'ru')
    return (ascending ? order : -order) || a.segment_id - b.segment_id
  }), [data.rows, pipe, sortKey, ascending])
  const pageCount = Math.ceil(rows.length / PAGE_SIZE)
  function sort(key: SortKey) {
    setAscending(sortKey === key ? !ascending : true)
    setSortKey(key)
    setPage(0)
  }
  function cell(row: RiskRow, key: SortKey) {
    if (key === 'p10_mm' || key === 'p90_mm') return <span className="muted">{format(row[key])}</span>
    if (key === 'risk_level') return <span className="risk-badge">В первую очередь</span>
    if (key === 'prediction_mm') return <strong>{format(row.prediction_mm)}</strong>
    if (key === 'segment_id') return <span className="mono">#{row.segment_id}</span>
    if (key === 'position_m') return integer(row.position_m)
    return <span className="mono pipe-id">{row.pipe_id}</span>
  }
  return <section id="risk" className="panel">
    <SectionHeading title="10 % самых опасных сегментов"
      description="Приоритет проверки по минимальной прогнозной толщине внутри каждой трубы."/>
    <div className="table-card"><div className="table-toolbar"><span className="toolbar-label">В списке: <strong>{integer(data.rows.length)}</strong></span>
      <label className="select-label">Трубопровод<select value={pipe} onChange={event => { setPipe(event.target.value); setPage(0) }} aria-label="Фильтр по трубе"><option value="all">Обе трубы</option>{pipeIds.map(id => <option key={id} value={id}>{id}</option>)}</select></label></div>
      <table className="risk-table"><caption className="sr-only">Самые тонкие сегменты по сохранённому прогнозу</caption><thead><tr>{columns.map(column => <th key={column.key} scope="col" aria-sort={sortKey === column.key ? ascending ? 'ascending' : 'descending' : 'none'}><button onClick={() => sort(column.key)}>{column.label}<span>{sortKey === column.key ? ascending ? '↑' : '↓' : '↕'}</span></button></th>)}<th scope="col"><span className="sr-only">Показать на профиле</span></th></tr></thead>
        <tbody>{rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(row => <tr key={`${row.pipe_id}-${row.segment_id}`}>
          {columns.map(column => <td key={column.key}>{cell(row, column.key)}</td>)}
          <td><button className="row-action" aria-label={`Показать сегмент ${row.segment_id} трубы ${row.pipe_id} на профиле`} onClick={() => onSelect({ pipeId: row.pipe_id, segmentId: row.segment_id })}>На профиле</button></td>
        </tr>)}</tbody>
      </table><div className="table-pagination"><span>{integer(page * PAGE_SIZE + 1)}–{integer(Math.min((page + 1) * PAGE_SIZE, rows.length))} из {integer(rows.length)} сегментов</span><div><button disabled={page === 0} onClick={() => setPage(page - 1)} aria-label="Предыдущая страница">←</button><span>Страница {page + 1} / {pageCount}</span><button disabled={page + 1 >= pageCount} onClick={() => setPage(page + 1)} aria-label="Следующая страница">→</button></div></div>
    </div><p className="footnote">P10–P90 — сохранённые сценарные границы. Они не калиброваны как предиктивный интервал.</p>
  </section>
}
