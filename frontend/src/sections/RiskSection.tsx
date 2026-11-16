import { Fragment, useMemo, useState } from 'react'
import { SectionHeading } from '../components/SectionHeading'
import { RiskSegmentDetails } from '../components/RiskSegmentDetails'
import type { ProfileData, RiskData, RiskRow, SegmentSelection } from '../types'
import { PAGE_SIZE } from '../utils/constants'
import { format, integer } from '../utils/format'

type SortKey = 'pipe_id' | 'segment_id' | 'position_m' | 'prediction_mm' | 'lower_mm' | 'upper_mm' | 'risk_level'
const columns: { key: SortKey; label: string }[] = [
  { key: 'pipe_id', label: 'Труба' }, { key: 'segment_id', label: 'Сегмент' },
  { key: 'position_m', label: 'Позиция, м' }, { key: 'prediction_mm', label: 'Прогноз, мм' },
  { key: 'lower_mm', label: 'Нижняя, мм' }, { key: 'upper_mm', label: 'Верхняя, мм' },
  { key: 'risk_level', label: 'Проверка' },
]

export function RiskSection({ data, profile, onSelect }: {
  data: RiskData
  profile: ProfileData
  onSelect: (selection: SegmentSelection) => void
}) {
  const [pipe, setPipe] = useState('all')
  const [sortKey, setSortKey] = useState<SortKey>('prediction_mm')
  const [ascending, setAscending] = useState(true)
  const [page, setPage] = useState(0)
  const [expandedKey, setExpandedKey] = useState<string | null>(null)
  const pipeIds = [...new Set(data.rows.map(row => row.pipe_id))]
  const rows = useMemo(() => data.rows.filter(row => pipe === 'all' || row.pipe_id === pipe).sort((a, b) => {
    const first = a[sortKey], second = b[sortKey]
    const order = typeof first === 'number' && typeof second === 'number' ? first - second : String(first).localeCompare(String(second), 'ru')
    return (ascending ? order : -order) || a.segment_id - b.segment_id
  }), [data.rows, pipe, sortKey, ascending])
  const pageCount = Math.ceil(rows.length / PAGE_SIZE)
  function toggleSegment(row: RiskRow) {
    const key = `${row.pipe_id}-${row.segment_id}`
    setExpandedKey(previous => previous === key ? null : key)
  }
  function changePage(nextPage: number) {
    setPage(nextPage)
    setExpandedKey(null)
  }
  function sort(key: SortKey) {
    setAscending(sortKey === key ? !ascending : true)
    setSortKey(key)
    setPage(0)
    setExpandedKey(null)
  }
  function cell(row: RiskRow, key: SortKey) {
    if (key === 'lower_mm' || key === 'upper_mm') return <span className="muted">{format(row[key])}</span>
    if (key === 'risk_level') return <span className="risk-badge">В первую очередь</span>
    if (key === 'prediction_mm') return <strong>{format(row.prediction_mm)}</strong>
    if (key === 'segment_id') {
      const expanded = expandedKey === `${row.pipe_id}-${row.segment_id}`
      return <button className="segment-toggle mono" type="button" aria-expanded={expanded}
        aria-controls={`segment-details-${row.pipe_id}-${row.segment_id}`}
        aria-label={`Сегмент ${row.segment_id}, труба ${row.pipe_id}`}
        onClick={() => toggleSegment(row)}>
        <span className={`segment-chevron${expanded ? ' is-open' : ''}`} aria-hidden="true">›</span>
        #{row.segment_id}
      </button>
    }
    if (key === 'position_m') return integer(row.position_m)
    return <span className="mono pipe-id">{row.pipe_id}</span>
  }
  return <section id="risk" className="panel">
    <SectionHeading title="10 % самых опасных сегментов"
      description="Приоритет по минимальной прогнозной толщине. Нажмите на номер сегмента, чтобы раскрыть подробности."/>
    <div className="table-card"><div className="table-toolbar"><span className="toolbar-label">В списке: <strong>{integer(data.rows.length)}</strong></span>
      <label className="select-label">Трубопровод<select value={pipe} onChange={event => { setPipe(event.target.value); changePage(0) }} aria-label="Фильтр по трубе"><option value="all">Обе трубы</option>{pipeIds.map(id => <option key={id} value={id}>{id}</option>)}</select></label></div>
      <table className="risk-table"><caption className="sr-only">Самые тонкие сегменты по сохранённому прогнозу</caption><thead><tr>{columns.map(column => <th key={column.key} scope="col" aria-sort={sortKey === column.key ? ascending ? 'ascending' : 'descending' : 'none'}><button onClick={() => sort(column.key)}>{column.label}<span>{sortKey === column.key ? ascending ? '↑' : '↓' : '↕'}</span></button></th>)}<th scope="col"><span className="sr-only">Подробности сегмента</span></th></tr></thead>
        <tbody>{rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map(row => {
          const key = `${row.pipe_id}-${row.segment_id}`
          const expanded = expandedKey === key
          const detailsId = `segment-details-${key}`
          return <Fragment key={key}>
            <tr className={expanded ? 'is-expanded' : undefined}>
              {columns.map(column => <td key={column.key}>{cell(row, column.key)}</td>)}
              <td><button className="row-action" type="button" aria-expanded={expanded} aria-controls={detailsId}
                aria-label={`Подробности сегмента ${row.segment_id}, труба ${row.pipe_id}`}
                onClick={() => toggleSegment(row)}>{expanded ? 'Свернуть' : 'Подробнее'}</button></td>
            </tr>
            {expanded && <tr className="segment-details-row" id={detailsId}>
              <td colSpan={columns.length + 1}>
                <RiskSegmentDetails row={row} pipe={profile.pipes.find(item => item.pipe_id === row.pipe_id)} onSelect={onSelect}/>
              </td>
            </tr>}
          </Fragment>
        })}</tbody>
      </table><div className="table-pagination"><span>{integer(page * PAGE_SIZE + 1)}–{integer(Math.min((page + 1) * PAGE_SIZE, rows.length))} из {integer(rows.length)} сегментов</span><div><button disabled={page === 0} onClick={() => changePage(page - 1)} aria-label="Предыдущая страница">←</button><span>Страница {page + 1} / {pageCount}</span><button disabled={page + 1 >= pageCount} onClick={() => changePage(page + 1)} aria-label="Следующая страница">→</button></div></div>
    </div><details className="method-details"><summary>Интервалы и ограничения приоритета</summary><p>{data.interval_note} Приоритет относительный: нормативная критическая толщина и вероятность отказа не заданы.</p></details>
  </section>
}
