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
      <div className="validation-details"><div className="metric-pair"><div><span>Pearson²</span><strong>{format(group.metrics.pearson_squared, 4)}</strong></div><div><span>R²</span><strong>{format(group.metrics.r2, 3)}</strong></div></div>
        <div className="detail-row"><span>MAE</span><strong>{format(group.metrics.mae_mm, 3)} мм</strong></div><div className="detail-row"><span>RMSE</span><strong>{format(group.metrics.rmse_mm, 3)} мм</strong></div>
        <p className="footnote">Построчные прогнозы доступны только для C0.</p>
        <details className="method-details"><summary>Схема валидации</summary><p>{group.description}</p></details>
      </div></div>
  </section>
}
