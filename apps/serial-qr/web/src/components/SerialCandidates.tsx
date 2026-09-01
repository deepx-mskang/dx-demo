import type { SerialCandidate } from '../lib/api'

interface Props {
  candidates: SerialCandidate[]
  selected: string | null
  onSelect: (serial: string) => void
}

export default function SerialCandidates({ candidates, selected, onSelect }: Props) {
  if (candidates.length === 0) return null

  return (
    <div className="space-y-2">
      <p className="dx-label">인식된 시리얼 후보</p>
      {candidates.map((c) => {
        const active = c.text === selected
        return (
          <button
            key={c.text}
            type="button"
            onClick={() => onSelect(c.text)}
            className={[
              'flex w-full items-center justify-between gap-3 rounded-xl border px-4 py-3 text-left transition-colors',
              active
                ? 'border-dx-cyan bg-dx-cyan/10'
                : 'border-dx-border bg-dx-surface hover:border-dx-cyanDim',
            ].join(' ')}
          >
            <span className="min-w-0">
              <span className="block font-mono text-base font-semibold">{c.text}</span>
              {c.normalized && (
                // 혼동 문자 보정(O→0 등)이 일어났음을 데모에서 설명할 수 있게 표시
                <span className="mt-0.5 block text-xs text-dx-amber">
                  혼동 문자 보정됨 · 원본 &ldquo;{c.rawText}&rdquo;
                </span>
              )}
            </span>
            <span className="shrink-0 font-mono text-sm text-dx-muted">
              {(c.score * 100).toFixed(1)}%
            </span>
          </button>
        )
      })}
    </div>
  )
}
