import type { ReactNode } from 'react'

export function SectionHeading({ title, description, aside }: {
  title: string; description?: string; aside?: ReactNode
}) {
  return <div className="section-heading">
    <div><h2>{title}</h2>{description && <p>{description}</p>}</div>
    {aside && <div className="heading-aside">{aside}</div>}
  </div>
}
