export const format = (value: number | null | undefined, digits = 2) =>
  value == null ? '—' : new Intl.NumberFormat('ru-RU', { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value)
export const integer = (value: number) => format(value, 0)
export const date = (value: string) => new Intl.DateTimeFormat('ru-RU', { timeZone: 'UTC' }).format(new Date(`${value.slice(0, 10)}T00:00:00Z`))
