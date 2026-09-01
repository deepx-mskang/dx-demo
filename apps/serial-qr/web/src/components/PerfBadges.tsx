import type { OcrPerf } from '../lib/api'

/**
 * NPU 추론 지연을 보여준다. 데모에서 "이게 NPU 로 돌고 있다"를
 * 설명하는 핵심 근거라 항상 노출한다.
 */
export default function PerfBadges({ perf }: { perf: OcrPerf }) {
  const items = [
    { label: '검출', value: `${perf.detMs.toFixed(1)} ms` },
    { label: '인식', value: `${perf.recMs.toFixed(1)} ms` },
    { label: '전체', value: `${perf.e2eMs.toFixed(1)} ms` },
    { label: '텍스트 박스', value: `${perf.numBoxes}` },
  ]

  return (
    <div className="flex flex-wrap gap-2">
      {items.map((it) => (
        <span
          key={it.label}
          className="rounded-lg border border-dx-border bg-dx-surface px-3 py-1.5 text-xs"
        >
          <span className="text-dx-muted">{it.label}</span>{' '}
          <span className="font-mono font-semibold text-dx-cyan">{it.value}</span>
        </span>
      ))}
      <span className="rounded-lg border border-dx-cyan/30 bg-dx-cyan/10 px-3 py-1.5 text-xs font-semibold text-dx-cyan">
        DX-M1 NPU
      </span>
    </div>
  )
}
