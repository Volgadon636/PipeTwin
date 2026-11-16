import { useState } from 'react'
import { SectionHeading } from '../components/SectionHeading'
import { Tabs } from '../components/Tabs'
import type { MetricsData } from '../types'
import { format, integer } from '../utils/format'

export function ComparisonSection({ data }: { data: MetricsData }) {
  const [cohort, setCohort] = useState(data.comparison_groups[0].id)
  const rows = data.comparison.filter(row => row.cohort === cohort)
  const group = data.comparison_groups.find(item => item.id === cohort) ?? data.comparison_groups[0]
  return <section id="comparison" className="panel comparison-panel">
    <SectionHeading title="Сравнение подходов"
      aside={<Tabs label="Сценарий сравнения" value={cohort} onChange={setCohort} options={data.comparison_groups.map(item => ({ id: item.id, label: item.label }))}/>}/>
    <div className="table-scroll"><table className="comparison-table"><caption className="sr-only">Сравнение сохранённых моделей</caption><thead><tr><th scope="col">Модель</th><th scope="col">N</th><th scope="col">Pearson r</th><th scope="col">Pearson²</th><th scope="col">MAE, мм</th><th scope="col">RMSE, мм</th><th scope="col">Смещение, мм</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.id} className={row.final ? 'final-model' : ''}><td><strong>{row.label}</strong>{row.final && <span className="model-badge">{cohort === 'S2_LOSO_pooled' ? 'Контроль H31' : 'В прогнозе'}</span>}</td><td>{integer(row.metrics.n)}</td><td>{format(row.metrics.pearson_r, 4)}</td><td>{format(row.metrics.pearson_squared, 4)}</td><td>{format(row.metrics.mae_mm, 3)}</td><td>{format(row.metrics.rmse_mm, 3)}</td><td>{format(row.metrics.bias_mm, 3)}</td></tr>)}</tbody></table></div>
    <p className="footnote">{group.description}</p>
    <details className="method-details"><summary>Условия сравнения и определение метрик</summary><p>{data.comparison_note} «—» — показатель не определён или не сохранён.</p></details>
  </section>
}
