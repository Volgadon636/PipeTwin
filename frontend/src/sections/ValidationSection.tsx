import { SectionHeading } from '../components/SectionHeading'
import { Tabs } from '../components/Tabs'
import { ValidationScatter } from '../charts/ValidationScatter'
import type { ValidationData, ValidationGroup } from '../types'
import { format, integer } from '../utils/format'

export function ValidationSection({ data, group, onChange }: { data: ValidationData; group: ValidationGroup; onChange: (id: string) => void }) {
  return <section id="validation" className="panel">
    <SectionHeading title="Прогноз — факт на валидации"
      description={`Модель ${group.model} · наблюдений: ${integer(group.metrics.n)}. Диагональ y = x — точное совпадение.`}
      aside={<Tabs label="Сценарий валидации" options={data.groups.map(item => ({ id: item.id, label: item.label }))} value={group.id} onChange={onChange}/>}/>
    <div className="validation-layout"><div className="chart-card"><ValidationScatter group={group}/></div>
      <div className="validation-details">{group.has_intervals ? <>
        <div className="metric-pair"><div><span>Покрытие</span><strong>{format((group.metrics.coverage ?? 0) * 100, 1)} %</strong></div><div><span>Заявленный уровень</span><strong>{format((data.nominal_coverage ?? 0) * 100, 0)} %</strong></div></div>
        <div className="detail-row"><span>Средняя ширина</span><strong>{format(group.metrics.mean_interval_width_mm, 3)} мм</strong></div>
        <p className="footnote">{group.metrics.coverage !== null && data.nominal_coverage !== null && group.metrics.coverage < data.nominal_coverage ? 'Заявленный уровень покрытия не подтверждён.' : 'Измеренное покрытие относится только к этой исторической проверке.'} Интервалы не гарантируют покрытие скрытой целевой трубы.</p>
      </> : <>
        <div className="metric-pair"><div><span>Pearson²</span><strong>{format(group.metrics.pearson_squared, 4)}</strong></div><div><span>R²</span><strong>{format(group.metrics.r2, 3)}</strong></div></div>
        <div className="detail-row"><span>Signed Pearson r</span><strong>{format(group.metrics.pearson_r, 4)}</strong></div>
        <div className="detail-row"><span>MAE</span><strong>{format(group.metrics.mae_mm, 3)} мм</strong></div><div className="detail-row"><span>RMSE</span><strong>{format(group.metrics.rmse_mm, 3)} мм</strong></div>
        <p className="footnote">Сохранённые прогнозы {group.model}. Сводная корреляция нескольких участков не является оценкой скрытой целевой трубы.</p>
      </>}
        <details className="method-details"><summary>Схема валидации</summary><p>{group.description}</p></details>
      </div></div>
    <details className="method-details"><summary>Все профили, включая малые группы</summary><div className="table-scroll"><table className="comparison-table"><thead><tr><th>Труба</th><th>N</th><th>Pearson r</th><th>Pearson²</th><th>Покрытие</th></tr></thead><tbody>{group.profiles.map(profile => <tr key={profile.pipe_id}><td>{profile.pipe_id}</td><td>{profile.metrics.n}</td><td>{format(profile.metrics.pearson_r, 4)}</td><td>{format(profile.metrics.pearson_squared, 4)}</td><td>{profile.metrics.coverage === null ? '—' : `${format(profile.metrics.coverage * 100, 1)} %`}</td></tr>)}</tbody></table></div><p>«—» — метрика не определена или не оценивалась. Малые группы не дают надёжного подтверждения качества.</p></details>
  </section>
}
