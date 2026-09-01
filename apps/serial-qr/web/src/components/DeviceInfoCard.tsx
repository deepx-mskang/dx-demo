import type { DeviceInfo, QaStatus } from '../data/devices'

const QA_STYLE: Record<QaStatus, string> = {
  PASS: 'bg-dx-green/15 text-dx-green border-dx-green/30',
  PENDING: 'bg-dx-amber/15 text-dx-amber border-dx-amber/30',
  FAIL: 'bg-dx-red/15 text-dx-red border-dx-red/30',
}

const QA_LABEL: Record<QaStatus, string> = {
  PASS: 'QA 합격',
  PENDING: 'QA 대기',
  FAIL: 'QA 불합격',
}

function Row({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-dx-border/60 py-2.5 last:border-0">
      <dt className="shrink-0 text-sm text-dx-muted">{label}</dt>
      <dd className={`text-right text-sm font-medium ${mono ? 'font-mono' : ''}`}>{value}</dd>
    </div>
  )
}

export default function DeviceInfoCard({ device }: { device: DeviceInfo }) {
  return (
    <div className="dx-card p-6">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="dx-label">시리얼 번호</p>
          <p className="font-mono text-xl font-bold tracking-wide text-dx-cyan">
            {device.serial}
          </p>
        </div>
        <span
          className={`rounded-full border px-3 py-1 text-xs font-semibold ${QA_STYLE[device.qaStatus]}`}
        >
          {QA_LABEL[device.qaStatus]}
        </span>
      </div>

      <dl>
        <Row label="모델" value={device.model} />
        <Row label="NPU" value={device.npu} />
        <Row label="HW 리비전" value={device.hwRevision} />
        <Row label="펌웨어" value={device.firmware} mono />
        <Row label="MAC 주소" value={device.macAddress} mono />
        <Row label="제조일" value={device.manufacturedAt} mono />
        <Row label="보증 만료" value={device.warrantyUntil} mono />
        <Row label="배치 위치" value={device.deployedSite} />
      </dl>

      <div className="mt-5 grid grid-cols-3 gap-3">
        {[
          { label: '연산', value: `${device.specs.tops}`, unit: 'TOPS' },
          { label: '메모리', value: `${device.specs.memoryGb}`, unit: 'GB' },
          { label: '소비전력', value: `${device.specs.powerW}`, unit: 'W' },
        ].map((s) => (
          <div key={s.label} className="rounded-xl border border-dx-border bg-dx-surface p-3">
            <p className="dx-label">{s.label}</p>
            <p className="mt-1">
              <span className="font-mono text-xl font-bold text-dx-text">{s.value}</span>
              <span className="ml-1 text-xs text-dx-muted">{s.unit}</span>
            </p>
          </div>
        ))}
      </div>
    </div>
  )
}
