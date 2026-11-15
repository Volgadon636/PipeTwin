import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { SectionHeading } from '../components/SectionHeading'
import { Tabs } from '../components/Tabs'
import { SceneBoundary } from '../components/SceneBoundary'
import type { ProfileData } from '../types'
import { date, format, integer } from '../utils/format'
import { PIPE_SCENE } from '../three/config'
import { clampSegmentIndex, getNextRiskIndex, getThickness, getThicknessColor, getThicknessRange, THICKNESS_STOPS } from '../three/pipeData'
import type { ThicknessSource } from '../three/pipeData'
import type { PipeViewMode } from '../three/types'

const PipeScene = lazy(() => import('../three/PipeScene'))

export function Pipe3DSection({ data }: { data: ProfileData }) {
  const section = useRef<HTMLElement>(null)
  const [enabled, setEnabled] = useState(false)
  const [pipeId, setPipeId] = useState(data.pipes[0].pipe_id)
  const [index, setIndex] = useState(0)
  const [mode, setMode] = useState<PipeViewMode>('outside')
  const [source, setSource] = useState<ThicknessSource>('prediction')
  const [playing, setPlaying] = useState(false)
  const [resetToken, setResetToken] = useState(0)
  const pipe = data.pipes.find(item => item.pipe_id === pipeId) ?? data.pipes[0]
  const currentIndex = clampSegmentIndex(pipe, index)
  const point = pipe.points[currentIndex]
  const range = useMemo(() => getThicknessRange(data.pipes, source), [data.pipes, source])
  const previousRisk = getNextRiskIndex(pipe, currentIndex, -1)
  const nextRisk = getNextRiskIndex(pipe, currentIndex, 1)
  const firstVisible = pipe.points[Math.max(0, currentIndex - PIPE_SCENE.visibleRadius)]
  const lastVisible = pipe.points[Math.min(pipe.points.length - 1, currentIndex + PIPE_SCENE.visibleRadius)]

  useEffect(() => {
    if (!section.current) return
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) { setEnabled(true); observer.disconnect() }
    }, { rootMargin: '180px' })
    observer.observe(section.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!playing) return
    const timer = window.setInterval(() => setIndex(value => Math.min(value + 1, pipe.points.length - 1)), PIPE_SCENE.autoStepMs)
    return () => window.clearInterval(timer)
  }, [playing, pipe.points.length])

  useEffect(() => {
    if (playing && currentIndex === pipe.points.length - 1) setPlaying(false)
  }, [currentIndex, pipe.points.length, playing])

  function move(nextIndex: number) { setPlaying(false); setIndex(clampSegmentIndex(pipe, nextIndex)) }
  function changePipe(id: string) { setPipeId(id); setIndex(0); setPlaying(false); setResetToken(value => value + 1) }

  return <section id="pipe-3d" className="panel pipe3d-panel" ref={section}>
    <SectionHeading title="3D-обзор трубы · эксперимент"
      description="Локальный участок вокруг выбранного сегмента. Цвет показывает минимальную толщину, а не форму дефекта."
      aside={<Tabs label="Труба в 3D" options={data.pipes.map(item => ({ id: item.pipe_id, label: item.pipe_id }))} value={pipeId} onChange={changePipe}/>}/>
    <div className="pipe3d-toolbar">
      <Tabs label="Режим 3D-просмотра" value={mode} onChange={value => { setMode(value as PipeViewMode); setPlaying(false) }}
        options={[{ id: 'outside', label: 'Снаружи' }, { id: 'inside', label: 'Внутри' }]}/>
      <label className="select-label">Окраска<select value={source} onChange={event => setSource(event.target.value as ThicknessSource)} aria-label="Окраска 3D-трубы">
        <option value="prediction">Прогноз толщины</option><option value="actual">Прошлое ВТД</option>
      </select></label>
      <button className="button" onClick={() => setResetToken(value => value + 1)}>Сбросить вид</button>
      <button className={`button${playing ? ' is-playing' : ''}`} disabled={currentIndex === pipe.points.length - 1} onClick={() => setPlaying(value => !value)}>{playing ? 'Пауза' : 'Автопроход'}</button>
    </div>
    <div className="pipe3d-layout">
      <div className="pipe3d-stage">
        {enabled ? <SceneBoundary key={pipeId}><Suspense fallback={<div className="pipe3d-fallback" role="status">Загрузка 3D-сцены…</div>}>
          <PipeScene key={pipeId} pipe={pipe} index={currentIndex} mode={mode} source={source} range={range} resetToken={resetToken} onIndexChange={move}/>
        </Suspense></SceneBoundary> : <div className="pipe3d-fallback"><button className="button" onClick={() => setEnabled(true)}>Открыть 3D-просмотр</button></div>}
        <div className="pipe3d-hud" aria-hidden="true"><span>{mode === 'outside' ? 'Осмотр снаружи' : 'Проход внутри'}</span><span>{integer(firstVisible.position_m)}–{integer(lastVisible.end_m)} м</span></div>
        <div className="pipe3d-help">{mode === 'outside' ? 'Тяните мышью — вращение · колесо — масштаб · клик — выбор сегмента' : 'Тяните мышью — осмотр · W/S или стрелки — движение при фокусе на сцене'}</div>
      </div>
      <aside className="pipe3d-inspector" aria-live="polite">
        <span className="pipe3d-kicker">Выбранный сегмент</span><h3>#{point.segment_id}</h3>
        <p>{integer(point.position_m)}–{format(point.end_m, 1)} м</p>
        <span className={point.is_risk ? 'risk-badge' : 'pipe3d-neutral'}>{point.is_risk ? 'Проверить в первую очередь' : 'Вне выбранных 10 %'}</span>
        <dl><div><dt>Прогноз</dt><dd>{format(point.prediction_mm)} мм</dd></div><div><dt>P10–P90</dt><dd>{format(point.p10_mm)}–{format(point.p90_mm)} мм</dd></div>
          <div><dt>Прошлое ВТД</dt><dd>{point.actual_mm === null ? 'Нет измерения' : `${format(point.actual_mm)} мм`}</dd></div></dl>
        <p className="pipe3d-date">Прогноз на {date(pipe.prediction_date)}{point.actual_date && <><br/>ВТД от {date(point.actual_date)}</>}</p>
        <div className="pipe3d-risk-navigation"><button className="button" disabled={previousRisk === null} onClick={() => { if (previousRisk !== null) move(previousRisk) }}>← К приоритетному</button><button className="button" disabled={nextRisk === null} onClick={() => { if (nextRisk !== null) move(nextRisk) }}>К приоритетному →</button></div>
      </aside>
    </div>
    <div className="pipe3d-navigation">
      <div className="pipe3d-navigation-heading"><strong>Позиция: {integer(point.position_m)} м</strong><span>Вся труба · {format(pipe.length_m / 1000)} км</span></div>
      <div className="pipe3d-strip" aria-hidden="true">{pipe.points.map(item => <span key={item.segment_id} style={{ background: getThicknessColor(getThickness(item, source), range) }}/>)}</div>
      <div className="pipe3d-slider"><button className="button" disabled={currentIndex === 0} onClick={() => move(currentIndex - 1)}>← Сегмент</button>
        <input type="range" min={0} max={pipe.points.length - 1} step={1} value={currentIndex} onChange={event => move(Number(event.target.value))} aria-label="Положение вдоль трубы в 3D" aria-valuetext={`Сегмент ${point.segment_id}, ${integer(point.position_m)} м`}/>
        <button className="button" disabled={currentIndex === pipe.points.length - 1} onClick={() => move(currentIndex + 1)}>Сегмент →</button></div>
    </div>
    <div className="pipe3d-color-legend"><span>Толщина, мм</span><span>{format(range?.min)}</span><i style={{ background: `linear-gradient(90deg, ${THICKNESS_STOPS.join(',')})` }}/><span>{format(range?.max)}</span><span className="pipe3d-missing-key">Нет измерения</span><span>Шкала общая для двух труб</span></div>
    <p className="footnote">Схематичная геометрия: длина и диаметр показаны в разных масштабах, толщина стенки геометрически не восстановлена. Кольцо соответствует сегменту сетки; цвет не указывает положение дефекта по окружности. Оранжевые кольца обозначают сохранённый приоритет проверки.</p>
  </section>
}
