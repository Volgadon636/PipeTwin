import { useState } from 'react'
import { useProjectData } from './hooks/useProjectData'
import { FactorsSection } from './sections/FactorsSection'
import { RiskSection } from './sections/RiskSection'
import { ValidationSection } from './sections/ValidationSection'
import { ProfileSection } from './sections/ProfileSection'
import { ComparisonSection } from './sections/ComparisonSection'
import { Pipe3DSection } from './sections/Pipe3DSection'
import type { FrontendData, SegmentSelection } from './types'

function Dashboard({ data }: { data: FrontendData }) {
  const [groupId, setGroupId] = useState(data.validation.groups[0].id)
  const [selection, setSelection] = useState<SegmentSelection | null>(null)
  const group = data.validation.groups.find(item => item.id === groupId) ?? data.validation.groups[0]
  function selectSegment(value: SegmentSelection) {
    setSelection(value)
    document.getElementById('profile')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  //return <> <Header/> </>
  return <> <main className="dashboard" aria-label="Метрики и прогноз толщины стенки">
    <div className="page-heading">
      {/* <div><h1>Прогноз толщины стенки</h1><p>{data.profile.pipes.length} трубы · {integer(data.metrics.segment_count)} сегментов · шаг {data.metrics.segment_size_m} м</p></div> */}
      {/* <span className="dataset-label">Модели v2 · {data.metrics.final_model}</span> */}
    </div>
    <ProfileSection data={data.profile} intervalMethod={data.validation.interval_method} selection={selection}/>
    <RiskSection data={data.risk} profile={data.profile} onSelect={selectSegment}/>
    <ValidationSection data={data.validation} group={group} onChange={setGroupId}/>
    <div className="overview-grid"><FactorsSection data={data.importance}/><ComparisonSection data={data.metrics}/></div>
    <Pipe3DSection data={data.profile}/>
  </main></>
}

export default function App() {
  const [retry, setRetry] = useState(0)
  const state = useProjectData(retry)
  if (state.status === 'loading') return <div className="status-screen" role="status"><span className="brand">PipeTwin</span><p>Загрузка данных…</p></div>
  if (state.status === 'error') return <div className="status-screen" role="alert"><span className="brand">PipeTwin</span><h1>Не удалось загрузить данные</h1><p>{state.message}</p><button className="button" onClick={() => setRetry(retry + 1)}>Повторить загрузку</button></div>
  return <Dashboard data={state.data}/>
}
