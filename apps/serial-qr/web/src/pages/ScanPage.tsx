import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CameraStream from '../components/CameraStream'
import Layout from '../components/Layout'
import PerfBadges from '../components/PerfBadges'
import SerialCandidates from '../components/SerialCandidates'
import { reasonToMessage, scan, type ScanResponse } from '../lib/api'
import { useDevice, useDevices } from '../hooks/useDevice'

export default function ScanPage() {
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<ScanResponse | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const runScan = useCallback(async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await scan()
      setResult(res)
      setSelected(res.serial)
      if (!res.ok) {
        setError(reasonToMessage(res.reason))
      }
    } catch (e) {
      setResult(null)
      setSelected(null)
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  const reset = useCallback(() => {
    setResult(null)
    setSelected(null)
    setError(null)
  }, [])

  // 스페이스바로도 스캔할 수 있게 한다 (데모 진행 중 마우스 이동 최소화)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'BUTTON'].includes(target.tagName)) return
      e.preventDefault()
      if (busy) return
      if (result) reset()
      else void runScan()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, result, reset, runScan])

  const { device: registered, loading: lookupLoading } = useDevice(selected ?? undefined)
  const { devices } = useDevices()

  return (
    <Layout
      step={1}
      title="시리얼 번호 인식"
      subtitle="기기 라벨을 카메라에 비추고 인식 버튼을 누르세요. PP-OCRv6 이 DX-M1 NPU 에서 동작합니다."
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <CameraStream frozenFrame={result?.frame ?? null} />

        <div className="flex flex-col gap-4">
          {!result ? (
            <>
              <button
                type="button"
                onClick={() => void runScan()}
                disabled={busy}
                className="dx-btn-primary w-full py-4 text-lg"
              >
                {busy ? '인식 중…' : '시리얼 인식'}
              </button>
              <p className="text-center text-xs text-dx-muted">
                스페이스바로도 실행할 수 있습니다.
              </p>
            </>
          ) : (
            <>
              <PerfBadges perf={result.perf} />

              <SerialCandidates
                candidates={result.candidates}
                selected={selected}
                onSelect={setSelected}
              />

              {selected && (
                <div className="dx-card p-4">
                  <p className="dx-label mb-1">선택된 시리얼</p>
                  <p className="font-mono text-xl font-bold text-dx-cyan">{selected}</p>
                  <p className="mt-2 text-sm">
                    {lookupLoading ? (
                      <span className="text-dx-muted">등록 여부 확인 중…</span>
                    ) : registered ? (
                      <span className="text-dx-green">
                        등록된 기기입니다 — {registered.model}
                      </span>
                    ) : (
                      <span className="text-dx-amber">
                        레지스트리에 없는 시리얼입니다.
                      </span>
                    )}
                  </p>

                  {!lookupLoading && !registered && (
                    <button
                      type="button"
                      onClick={() =>
                        navigate(`/register?serial=${encodeURIComponent(selected)}`)
                      }
                      className="dx-btn-primary mt-3 w-full py-2.5 text-sm"
                    >
                      이 기기 등록하기 →
                    </button>
                  )}
                </div>
              )}

              {result.rawTexts.length > 0 && (
                <details className="dx-card p-4">
                  <summary className="cursor-pointer text-sm text-dx-muted">
                    OCR 원본 텍스트 {result.rawTexts.length}건
                  </summary>
                  <ul className="mt-3 space-y-1">
                    {result.rawTexts.map((t, i) => (
                      <li key={`${t}-${i}`} className="font-mono text-sm text-dx-text">
                        {t}
                      </li>
                    ))}
                  </ul>
                </details>
              )}

              <div className="mt-auto flex gap-3">
                <button type="button" onClick={reset} className="dx-btn-ghost flex-1">
                  다시 스캔
                </button>
                <button
                  type="button"
                  disabled={!selected}
                  onClick={() => selected && navigate(`/result/${encodeURIComponent(selected)}`)}
                  className="dx-btn-primary flex-1"
                >
                  QR 생성 →
                </button>
              </div>
            </>
          )}

          {error && (
            <div className="rounded-xl border border-dx-red/40 bg-dx-red/10 p-4 text-sm text-dx-red">
              {error}
            </div>
          )}

          <button
            type="button"
            onClick={() => navigate('/register')}
            className="dx-btn-ghost w-full"
          >
            + 기기 사전 등록
          </button>

          <details className="dx-card p-4">
            <summary className="cursor-pointer text-sm text-dx-muted">
              라벨 없이 시연하기 (등록된 시리얼 직접 선택 · {devices.length}대)
            </summary>
            <div className="mt-3 flex flex-wrap gap-2">
              {devices.map((d) => (
                <button
                  key={d.serial}
                  type="button"
                  onClick={() => navigate(`/result/${encodeURIComponent(d.serial)}`)}
                  className="rounded-lg border border-dx-border bg-dx-surface px-3 py-1.5 font-mono text-xs hover:border-dx-cyanDim"
                >
                  {d.serial}
                </button>
              ))}
            </div>
          </details>
        </div>
      </div>
    </Layout>
  )
}
