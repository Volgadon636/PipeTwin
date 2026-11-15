export function Tabs({ label, options, value, onChange }: {
  label: string; options: { id: string; label: string }[]; value: string; onChange: (value: string) => void
}) {
  return <div className="tabs" role="group" aria-label={label}>
    {options.map(option => <button key={option.id} type="button" aria-pressed={value === option.id}
      className={value === option.id ? 'active' : ''} onClick={() => onChange(option.id)}>{option.label}</button>)}
  </div>
}
