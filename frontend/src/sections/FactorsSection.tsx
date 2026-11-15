import { SectionHeading } from '../components/SectionHeading'
import type { ImportanceData } from '../types'

export function FactorsSection({ data }: { data: ImportanceData }) {
  return <section id="factors" className="panel factors-panel">
    <SectionHeading title="Влияние факторов"/>
    <p className="data-note">Числовая важность и SHAP не сохранены. Ранжировать факторы по силе влияния нельзя.</p>
    <h3 className="list-heading">Учтены в модели {data.physical_model}</h3>
    <ul className="factor-list">{data.known_factors.map(factor => <li key={factor}>{factor}</li>)}</ul>
  </section>
}
