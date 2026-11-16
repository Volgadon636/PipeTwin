import { useState } from 'react'
import { SectionHeading } from '../components/SectionHeading'
import { Tabs } from '../components/Tabs'
import type { MetricsData } from '../types'
import { format, integer } from '../utils/format'

export function ComparisonSection({ data }: { data: MetricsData }) {
  const [cohort, setCohort] = useState('cold_start')
  const rows = data.comparison.filter(row => row.cohort === cohort)
  return <section id="comparison" className="panel comparison-panel">
    <SectionHeading title="Сравнение подходов"
      aside={<Tabs label="Сценарий сравнения" value={cohort} onChange={setCohort} options={[{ id: 'cold_start', label: 'Холодный старт' }, { id: 'between_vtd', label: 'Между ВТД' }]}/>}/>
    <div className="table-scroll"><table className="comparison-table"><caption className="sr-only">Сравнение сохранённых моделей</caption><thead><tr><th scope="col">Модель</th><th scope="col">N</th><th scope="col">Pearson²</th><th scope="col">MAE, мм</th><th scope="col">RMSE, мм</th><th scope="col">Смещение, мм</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.id} className={row.final ? 'final-model' : ''}><td><strong>{row.label}</strong>{row.final && <span className="model-badge">В прогнозе</span>}</td><td>{integer(row.metrics.n)}</td><td>{format(row.metrics.pearson_squared, 4)}</td><td>{format(row.metrics.mae_mm, 3)}</td><td>{format(row.metrics.rmse_mm, 3)}</td><td>{format(row.metrics.bias_mm, 3)}</td></tr>)}</tbody></table></div>
    <details className="method-details"><summary>Условия сравнения и определение метрик</summary><p>{data.comparison_note} «—» — показатель не определён или не сохранён.</p></details>
  </section>
}
