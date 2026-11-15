import { Component } from 'react'
import type { ReactNode } from 'react'

export class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() { return { failed: true } }

  render() {
    return this.state.failed
      ? <div className="pipe3d-fallback" role="status"><strong>3D-просмотр не загрузился</strong><p>Обновите страницу, чтобы повторить попытку. Данные сегментов доступны справа и в таблице.</p></div>
      : this.props.children
  }
}
