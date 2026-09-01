import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import Layout from '../components/Layout'
import { registerDevice } from '../lib/api'
import { useDevices } from '../hooks/useDevice'
import {
  MODEL_OPTIONS,
  QA_OPTIONS,
  SITE_EXAMPLES,
  makeExampleDraft,
  type DeviceInfo,
  type QaStatus,
} from '../data/devices'

/**
 * 기기 사전 등록.
 *
 * 두 가지 경로로 들어온다.
 *  - `/register`               사전 등록 버튼 → 시리얼은 비워 두고 나머지는 예시로 프리필
 *  - `/register?serial=XXX`    스캔에서 넘어옴 → 인식된 시리얼을 채우고 나머지는 예시로 프리필
 *
 * 시리얼은 유니크하다. 중복은 서버가 409 로 막고, 여기서도 미리 걸러 준다.
 */
export default function RegisterPage() {
  const [params] = useSearchParams()
  const navigate = useNavigate()
  const scannedSerial = (params.get('serial') ?? '').trim().toUpperCase()
  const fromScan = scannedSerial.length > 0

  const { devices, loading: devicesLoading, reload } = useDevices()
  const [draft, setDraft] = useState<DeviceInfo>(() => makeExampleDraft(scannedSerial))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const serialRef = useRef<HTMLInputElement>(null)

  // 스캔에서 넘어온 시리얼이 바뀌면 폼을 다시 만든다.
  useEffect(() => {
    setDraft(makeExampleDraft(scannedSerial))
    setError(null)
  }, [scannedSerial])

  // 사전 등록은 시리얼부터 입력받는다.
  useEffect(() => {
    if (!fromScan) serialRef.current?.focus()
  }, [fromScan])

  const serial = draft.serial.trim().toUpperCase()

  const duplicate = useMemo(
    () => devices.find((d) => d.serial.toUpperCase() === serial) ?? null,
    [devices, serial],
  )

  const serialValid = /^[A-Z0-9][A-Z0-9-]{2,30}[A-Z0-9]$/.test(serial)
  const canSubmit = serialValid && !duplicate && draft.model.trim().length > 0 && !busy

  const set = <K extends keyof DeviceInfo>(key: K, value: DeviceInfo[K]) =>
    setDraft((d) => ({ ...d, [key]: value }))

  const setSpec = (key: keyof DeviceInfo['specs'], value: number) =>
    setDraft((d) => ({ ...d, specs: { ...d.specs, [key]: value } }))

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!canSubmit) return

    setBusy(true)
    setError(null)
    try {
      const outcome = await registerDevice({ ...draft, serial })
      if (!outcome.ok) {
        setError(outcome.message)
        if (outcome.reason === 'duplicate_serial') reload()
        return
      }
      navigate(`/result/${encodeURIComponent(serial)}`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Layout
      title={fromScan ? '인식된 기기 등록' : '기기 사전 등록'}
      subtitle={
        fromScan
          ? '카메라로 읽은 시리얼입니다. 나머지 항목은 예시로 채워져 있으니 필요한 것만 고치세요.'
          : '시리얼 번호만 입력하면 됩니다. 나머지 항목은 예시로 채워져 있습니다.'
      }
    >
      <form onSubmit={submit} className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-6">
          {/* --- 시리얼 --- */}
          <section className="dx-card p-6">
            <label htmlFor="serial" className="dx-label">
              시리얼 번호 <span className="text-dx-cyan">*</span>
            </label>

            <input
              id="serial"
              ref={serialRef}
              value={draft.serial}
              onChange={(e) => set('serial', e.target.value.toUpperCase())}
              readOnly={fromScan}
              placeholder="DX-M1-A7K3P9V2"
              spellCheck={false}
              autoComplete="off"
              className={[
                'mt-2 w-full rounded-xl border bg-dx-surface px-4 py-3 font-mono text-xl',
                'tracking-wide outline-none transition-colors placeholder:text-dx-muted/50',
                fromScan ? 'cursor-default border-dx-cyan/40 text-dx-cyan' : 'border-dx-border',
                duplicate ? 'border-dx-red' : '',
              ].join(' ')}
            />

            {fromScan ? (
              <p className="mt-2 text-xs text-dx-cyan">
                카메라로 인식된 시리얼입니다.{' '}
                <Link to="/register" className="underline">
                  직접 입력하려면 여기
                </Link>
              </p>
            ) : (
              <p className="mt-2 text-xs text-dx-muted">
                영문·숫자·하이픈 4~32자. 예: <code className="font-mono">DX-M1-A7K3P9V2</code>
              </p>
            )}

            {serial.length > 0 && !serialValid && (
              <p className="mt-2 text-sm text-dx-red">
                형식이 올바르지 않습니다. 영문·숫자·하이픈 4~32자여야 합니다.
              </p>
            )}

            {duplicate && (
              <p className="mt-2 text-sm text-dx-red">
                이미 등록된 시리얼입니다 ({duplicate.deployedSite}).{' '}
                <Link
                  to={`/device/${encodeURIComponent(duplicate.serial)}`}
                  className="underline"
                >
                  등록 정보 보기
                </Link>
              </p>
            )}
          </section>

          {/* --- 기기 정보 --- */}
          <section className="dx-card space-y-4 p-6">
            <h2 className="text-sm font-semibold text-dx-muted">기기 정보</h2>

            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="모델">
                <select
                  value={draft.model}
                  onChange={(e) => set('model', e.target.value)}
                  className="dx-input"
                >
                  {MODEL_OPTIONS.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="NPU">
                <input
                  value={draft.npu}
                  onChange={(e) => set('npu', e.target.value)}
                  className="dx-input"
                />
              </Field>

              <Field label="HW 리비전">
                <input
                  value={draft.hwRevision}
                  onChange={(e) => set('hwRevision', e.target.value)}
                  className="dx-input"
                />
              </Field>

              <Field label="펌웨어">
                <input
                  value={draft.firmware}
                  onChange={(e) => set('firmware', e.target.value)}
                  className="dx-input font-mono"
                />
              </Field>

              <Field label="MAC 주소">
                <input
                  value={draft.macAddress}
                  onChange={(e) => set('macAddress', e.target.value)}
                  className="dx-input font-mono"
                />
              </Field>

              <Field label="QA 상태">
                <select
                  value={draft.qaStatus}
                  onChange={(e) => set('qaStatus', e.target.value as QaStatus)}
                  className="dx-input"
                >
                  {QA_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>
                      {o.label}
                    </option>
                  ))}
                </select>
              </Field>

              <Field label="제조일">
                <input
                  type="date"
                  value={draft.manufacturedAt}
                  onChange={(e) => set('manufacturedAt', e.target.value)}
                  className="dx-input font-mono"
                />
              </Field>

              <Field label="보증 만료">
                <input
                  type="date"
                  value={draft.warrantyUntil}
                  onChange={(e) => set('warrantyUntil', e.target.value)}
                  className="dx-input font-mono"
                />
              </Field>
            </div>

            <Field label="배치 위치">
              <input
                value={draft.deployedSite}
                onChange={(e) => set('deployedSite', e.target.value)}
                list="site-examples"
                className="dx-input"
              />
              <datalist id="site-examples">
                {SITE_EXAMPLES.map((s) => (
                  <option key={s} value={s} />
                ))}
              </datalist>
            </Field>

            <div className="grid grid-cols-3 gap-4">
              <Field label="연산 (TOPS)">
                <input
                  type="number"
                  value={draft.specs.tops}
                  onChange={(e) => setSpec('tops', Number(e.target.value))}
                  className="dx-input font-mono"
                />
              </Field>
              <Field label="메모리 (GB)">
                <input
                  type="number"
                  value={draft.specs.memoryGb}
                  onChange={(e) => setSpec('memoryGb', Number(e.target.value))}
                  className="dx-input font-mono"
                />
              </Field>
              <Field label="소비전력 (W)">
                <input
                  type="number"
                  value={draft.specs.powerW}
                  onChange={(e) => setSpec('powerW', Number(e.target.value))}
                  className="dx-input font-mono"
                />
              </Field>
            </div>
          </section>
        </div>

        {/* --- 사이드 --- */}
        <aside className="flex flex-col gap-4">
          <div className="dx-card p-5">
            <p className="dx-label">등록 후</p>
            <p className="mt-2 text-sm text-dx-muted">
              등록이 끝나면 바로 QR 발행 화면으로 이동합니다. 그 QR 을 휴대폰으로 찍으면
              지금 입력한 정보가 그대로 조회됩니다.
            </p>
          </div>

          <button type="submit" disabled={!canSubmit} className="dx-btn-primary w-full py-4">
            {busy ? '등록 중…' : '등록하고 QR 발행 →'}
          </button>

          <button
            type="button"
            onClick={() => (fromScan ? navigate('/') : navigate(-1))}
            className="dx-btn-ghost w-full"
          >
            취소
          </button>

          {error && (
            <div className="rounded-xl border border-dx-red/40 bg-dx-red/10 p-4 text-sm text-dx-red">
              {error}
            </div>
          )}

          <div className="dx-card p-5">
            <p className="dx-label">현재 등록된 기기</p>
            <p className="mt-1 font-mono text-2xl font-bold text-dx-cyan">
              {devicesLoading ? '—' : devices.length}
              <span className="ml-1 text-sm font-normal text-dx-muted">대</span>
            </p>
          </div>
        </aside>
      </form>
    </Layout>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="dx-label">{label}</span>
      <div className="mt-1.5">{children}</div>
    </label>
  )
}
