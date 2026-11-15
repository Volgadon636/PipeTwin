import { useEffect, useRef, useState } from 'react'
import { createPipeScene } from './createPipeScene'
import type { PipeSceneController, PipeSceneProps } from './types'

export default function PipeScene(props: PipeSceneProps) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const controller = useRef<PipeSceneController | null>(null)
  const latest = useRef(props)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    latest.current = props
    controller.current?.update(props)
  }, [props])

  useEffect(() => {
    if (!canvas.current || error) return
    try {
      const scene = createPipeScene(canvas.current, props.pipe, latest.current,
        index => latest.current.onIndexChange(index),
        () => setError('Соединение с 3D-сценой потеряно. Перезагрузите страницу, чтобы попробовать снова.'))
      controller.current = scene
      return () => { controller.current = null; scene.dispose() }
    } catch {
      setError('Браузер не смог запустить WebGL. Данные сегментов доступны справа и в таблице выше.')
    }
  }, [props.pipe, error])

  if (error) return <div className="pipe3d-fallback" role="status"><strong>3D-просмотр недоступен</strong><p>{error}</p></div>
  return <canvas ref={canvas} className="pipe3d-canvas" tabIndex={0}
    aria-label="Интерактивная 3D-труба. Мышь — осмотр, W и S или стрелки — движение по сегментам."
    onKeyDown={event => {
      if (event.altKey || event.ctrlKey || event.metaKey) return
      const key = event.key.toLowerCase()
      const direction = ['w', 'arrowup', 'arrowright'].includes(key) ? 1 : ['s', 'arrowdown', 'arrowleft'].includes(key) ? -1 : 0
      if (direction) { event.preventDefault(); props.onIndexChange(props.index + direction) }
    }}/>
}
