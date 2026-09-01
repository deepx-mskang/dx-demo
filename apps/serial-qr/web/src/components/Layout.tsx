import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'

interface Props {
  title: string
  subtitle?: string
  step?: 1 | 2 | 3
  children: ReactNode
}

const STEPS = ['시리얼 인식', 'QR 발행', '기기 조회']

export default function Layout({ title, subtitle, step, children }: Props) {
  return (
    <div className="min-h-full bg-dx-bg">
      <header className="border-b border-dx-border bg-dx-surface/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-3 px-6 py-4">
          <Link to="/" className="flex items-baseline gap-2">
            <span className="text-lg font-bold tracking-tight text-dx-cyan">DEEPX</span>
            <span className="text-sm text-dx-muted">Serial-QR Demo</span>
          </Link>

          {step && (
            <ol className="ml-auto flex items-center gap-2 text-xs">
              {STEPS.map((label, i) => {
                const n = (i + 1) as 1 | 2 | 3
                const active = n === step
                const done = n < step
                return (
                  <li key={label} className="flex items-center gap-2">
                    <span
                      className={[
                        'flex items-center gap-2 rounded-full px-3 py-1.5 font-medium',
                        active
                          ? 'bg-dx-cyan text-dx-bg'
                          : done
                            ? 'bg-dx-cyan/15 text-dx-cyan'
                            : 'bg-dx-card text-dx-muted',
                      ].join(' ')}
                    >
                      <span className="tabular-nums">{done ? '✓' : n}</span>
                      {label}
                    </span>
                    {n < 3 && <span className="text-dx-border">→</span>}
                  </li>
                )
              })}
            </ol>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-6">
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="mt-1 text-sm text-dx-muted">{subtitle}</p>}
        </div>
        {children}
      </main>
    </div>
  )
}
