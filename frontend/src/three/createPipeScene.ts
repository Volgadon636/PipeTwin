import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import type { Pipe } from '../types'
import { PIPE_SCENE } from './config'
import { getThickness, getThicknessColor } from './pipeData'
import type { PipeSceneController, PipeSceneView } from './types'

export function createPipeScene(
  canvas: HTMLCanvasElement,
  pipe: Pipe,
  initialView: PipeSceneView,
  onIndexChange: (index: number) => void,
  onContextLost: () => void,
): PipeSceneController {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'low-power' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, PIPE_SCENE.pixelRatioLimit))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.95

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(PIPE_SCENE.background)
  scene.fog = new THREE.Fog(PIPE_SCENE.background, 25, 70)
  const camera = new THREE.PerspectiveCamera(48, 1, 0.025, 120)
  scene.add(camera)
  scene.add(new THREE.HemisphereLight('#d5e6ff', '#26394b', 1.2))
  const sunlight = new THREE.DirectionalLight('#fff2dc', 2)
  sunlight.position.set(-8, 12, -10)
  scene.add(sunlight)
  const fill = new THREE.DirectionalLight('#7fbcdd', 1)
  fill.position.set(8, 2, 8)
  scene.add(fill)
  const headlamp = new THREE.PointLight('#fff0d8', 7, 24, 2)
  camera.add(headlamp)

  const controls = new OrbitControls(camera, canvas)
  controls.enableDamping = true
  controls.enablePan = false
  controls.minDistance = 6
  controls.maxDistance = 65
  controls.minPolarAngle = 0.08
  controls.maxPolarAngle = Math.PI - 0.08

  const capacity = PIPE_SCENE.visibleRadius * 2 + 1
  const outerGeometry = new THREE.CylinderGeometry(PIPE_SCENE.outerRadius, PIPE_SCENE.outerRadius, 1, PIPE_SCENE.radialSegments, 1, true)
  const innerGeometry = new THREE.CylinderGeometry(PIPE_SCENE.innerRadius, PIPE_SCENE.innerRadius, 1, PIPE_SCENE.radialSegments, 1, true)
  outerGeometry.rotateX(Math.PI / 2)
  innerGeometry.rotateX(Math.PI / 2)
  const outerMaterial = new THREE.MeshStandardMaterial({ roughness: 0.58, metalness: 0.22 })
  const innerMaterial = new THREE.MeshStandardMaterial({ roughness: 0.75, metalness: 0.08, side: THREE.BackSide })
  const outer = new THREE.InstancedMesh(outerGeometry, outerMaterial, capacity)
  const inner = new THREE.InstancedMesh(innerGeometry, innerMaterial, capacity)
  const seamMaterial = new THREE.MeshBasicMaterial({ color: '#ffffff' })
  const seams = new THREE.InstancedMesh(new THREE.TorusGeometry(PIPE_SCENE.outerRadius + 0.008, 0.012, 6, 48), seamMaterial, capacity)
  const innerSeams = new THREE.InstancedMesh(new THREE.TorusGeometry(PIPE_SCENE.innerRadius - 0.006, 0.01, 6, 48), seamMaterial, capacity)
  const pipeGroup = new THREE.Group()
  pipeGroup.add(outer, inner, seams, innerSeams)
  scene.add(pipeGroup)
  for (const mesh of [outer, inner, seams, innerSeams]) {
    mesh.frustumCulled = false
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
  }

  const capGeometry = new THREE.RingGeometry(PIPE_SCENE.innerRadius, PIPE_SCENE.outerRadius, PIPE_SCENE.radialSegments)
  const caps = [0, 1].map(() => new THREE.Mesh(capGeometry, new THREE.MeshStandardMaterial({ side: THREE.DoubleSide, roughness: 0.65 })))
  pipeGroup.add(...caps)
  const selectionMaterial = new THREE.MeshBasicMaterial({ color: '#fff1ce' })
  const outerSelectionGeometry = new THREE.TorusGeometry(PIPE_SCENE.outerRadius + 0.035, 0.025, 8, 64)
  const innerSelectionGeometry = new THREE.TorusGeometry(PIPE_SCENE.innerRadius - 0.02, 0.018, 8, 64)
  const selection = [outerSelectionGeometry, outerSelectionGeometry, innerSelectionGeometry, innerSelectionGeometry]
    .map(geometry => new THREE.Mesh(geometry, selectionMaterial))
  pipeGroup.add(...selection)
  const grid = new THREE.GridHelper(120, 60, '#30485c', '#1b3041')
  grid.position.y = -3.1
  for (const material of Array.isArray(grid.material) ? grid.material : [grid.material]) {
    material.transparent = true
    material.opacity = 0.4
  }
  scene.add(grid)

  let current = initialView
  let targetZ = 0
  let dirty = true
  let visible = true
  let disposed = false
  let frame = 0
  let yaw = 0
  let pitch = 0
  let segmentIndices: number[] = []
  let pointerStart: { x: number; y: number; lastX: number; lastY: number; dragged: boolean } | null = null
  const transform = new THREE.Object3D()
  const color = new THREE.Color()
  const raycaster = new THREE.Raycaster()
  const pointer = new THREE.Vector2()

  function invalidate() { dirty = true }
  controls.addEventListener('change', invalidate)

  function setCamera() {
    controls.enableDamping = false
    controls.update()
    controls.enableDamping = true
    controls.enabled = current.mode === 'outside'
    grid.visible = current.mode === 'outside'
    headlamp.visible = current.mode === 'inside'
    camera.fov = current.mode === 'inside' ? 82 : 48
    camera.updateProjectionMatrix()
    yaw = 0
    pitch = 0
    if (current.mode === 'outside') {
      camera.position.set(12, 8, -22)
      controls.target.set(0, 0, 0)
      controls.update()
    } else {
      camera.position.set(0, 0, -0.1)
      camera.lookAt(0, 0, 10)
    }
    invalidate()
  }

  function update(view: PipeSceneView) {
    const resetCamera = view.mode !== current.mode || view.resetToken !== current.resetToken
    const point = pipe.points[view.index]
    if (!point) return
    current = view
    const start = Math.max(0, view.index - PIPE_SCENE.visibleRadius)
    const end = Math.min(pipe.points.length, view.index + PIPE_SCENE.visibleRadius + 1)
    segmentIndices = []
    for (let index = start; index < end; index++) {
      const item = pipe.points[index]
      const instance = index - start
      segmentIndices.push(index)
      transform.position.set(0, 0, (item.position_m + item.end_m) / 2 * PIPE_SCENE.metersToWorld)
      transform.scale.set(1, 1, (item.end_m - item.position_m) * PIPE_SCENE.metersToWorld)
      transform.updateMatrix()
      outer.setMatrixAt(instance, transform.matrix)
      inner.setMatrixAt(instance, transform.matrix)
      color.set(getThicknessColor(getThickness(item, view.source), view.range))
      outer.setColorAt(instance, color)
      inner.setColorAt(instance, color)
      transform.position.z = item.position_m * PIPE_SCENE.metersToWorld
      transform.scale.set(1, 1, 1)
      transform.updateMatrix()
      seams.setMatrixAt(instance, transform.matrix)
      innerSeams.setMatrixAt(instance, transform.matrix)
      color.set(item.is_risk ? '#e8a65c' : '#506c80')
      seams.setColorAt(instance, color)
      innerSeams.setColorAt(instance, color)
    }
    for (const mesh of [outer, inner, seams, innerSeams]) {
      mesh.count = end - start
      mesh.instanceMatrix.needsUpdate = true
      if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
    }
    outer.computeBoundingSphere()
    inner.computeBoundingSphere()
    const first = pipe.points[start]
    const last = pipe.points[end - 1]
    caps[0].position.z = first.position_m * PIPE_SCENE.metersToWorld
    caps[1].position.z = last.end_m * PIPE_SCENE.metersToWorld
    caps[0].material.color.set(getThicknessColor(getThickness(first, view.source), view.range))
    caps[1].material.color.set(getThicknessColor(getThickness(last, view.source), view.range))
    selection.forEach((marker, index) => {
      marker.position.z = (index % 2 === 0 ? point.position_m : point.end_m) * PIPE_SCENE.metersToWorld
    })
    const center = view.mode === 'inside'
      ? (point.position_m + point.end_m) / 2
      : (first.position_m + last.end_m) / 2
    targetZ = -center * PIPE_SCENE.metersToWorld
    if (Math.abs(pipeGroup.position.z - targetZ) > 6 || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      pipeGroup.position.z = targetZ
    }
    if (resetCamera) setCamera()
    invalidate()
  }

  function resize() {
    const width = canvas.clientWidth
    const height = canvas.clientHeight
    if (width === 0 || height === 0) return
    renderer.setSize(width, height, false)
    camera.aspect = width / height
    camera.updateProjectionMatrix()
    invalidate()
  }
  const resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(canvas)
  const intersectionObserver = new IntersectionObserver(entries => {
    visible = entries[0]?.isIntersecting ?? false
    if (visible) invalidate()
  })
  intersectionObserver.observe(canvas)

  function pointerDown(event: PointerEvent) {
    if (event.button !== 0) return
    pointerStart = { x: event.clientX, y: event.clientY, lastX: event.clientX, lastY: event.clientY, dragged: false }
    if (current.mode === 'inside') canvas.setPointerCapture(event.pointerId)
  }
  function pointerMove(event: PointerEvent) {
    if (!pointerStart) return
    if (Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) pointerStart.dragged = true
    if (current.mode !== 'inside') return
    yaw -= (event.clientX - pointerStart.lastX) * 0.004
    pitch = THREE.MathUtils.clamp(pitch - (event.clientY - pointerStart.lastY) * 0.004, -1.25, 1.25)
    pointerStart.lastX = event.clientX
    pointerStart.lastY = event.clientY
    camera.lookAt(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), camera.position.z + Math.cos(yaw) * Math.cos(pitch))
    invalidate()
  }
  function pointerUp(event: PointerEvent) {
    if (!pointerStart) return
    const moved = pointerStart.dragged || Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5
    pointerStart = null
    if (current.mode === 'inside' && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
    if (moved) return
    const bounds = canvas.getBoundingClientRect()
    pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -(event.clientY - bounds.top) / bounds.height * 2 + 1)
    raycaster.setFromCamera(pointer, camera)
    const hit = raycaster.intersectObjects([outer, inner], false)[0]
    if (hit?.instanceId !== undefined) {
      const index = segmentIndices[hit.instanceId]
      if (index !== undefined) onIndexChange(index)
    }
  }
  function pointerCancel() { pointerStart = null }
  function contextLost(event: Event) { event.preventDefault(); onContextLost() }
  canvas.addEventListener('pointerdown', pointerDown)
  canvas.addEventListener('pointermove', pointerMove)
  canvas.addEventListener('pointerup', pointerUp)
  canvas.addEventListener('pointercancel', pointerCancel)
  canvas.addEventListener('webglcontextlost', contextLost)

  update(initialView)
  setCamera()
  resize()
  function animate() {
    if (disposed) return
    if (visible) {
      const moving = Math.abs(pipeGroup.position.z - targetZ) > 0.001
      if (moving) pipeGroup.position.z = THREE.MathUtils.lerp(pipeGroup.position.z, targetZ, 0.18)
      const orbiting = current.mode === 'outside' && controls.update()
      if (dirty || moving || orbiting) {
        renderer.render(scene, camera)
        dirty = false
      }
    }
    frame = requestAnimationFrame(animate)
  }
  animate()

  return {
    update,
    dispose() {
      disposed = true
      cancelAnimationFrame(frame)
      resizeObserver.disconnect()
      intersectionObserver.disconnect()
      controls.removeEventListener('change', invalidate)
      controls.dispose()
      canvas.removeEventListener('pointerdown', pointerDown)
      canvas.removeEventListener('pointermove', pointerMove)
      canvas.removeEventListener('pointerup', pointerUp)
      canvas.removeEventListener('pointercancel', pointerCancel)
      canvas.removeEventListener('webglcontextlost', contextLost)
      const resources = new Set<THREE.BufferGeometry | THREE.Material>()
      scene.traverse(object => {
        if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
          resources.add(object.geometry)
          for (const material of Array.isArray(object.material) ? object.material : [object.material]) resources.add(material)
        }
      })
      resources.forEach(resource => resource.dispose())
      for (const mesh of [outer, inner, seams, innerSeams]) mesh.dispose()
      renderer.dispose()
    },
  }
}
