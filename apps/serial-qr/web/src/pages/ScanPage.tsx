import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import CameraStream from '../components/CameraStream'
import Layout from '../components/Layout'
import PerfBadges from '../components/PerfBadges'
import SerialCandidates from '../components/SerialCandidates'
import { autoReasonToMessage, reasonToMessage, scan, type ScanResponse } from '../lib/api'
import { useDevice, useDevices } from '../hooks/useDevice'

/** 실시간 스캔 간격. OCR 자체가 ~200ms 라 사실상 연속으로 돈다. */
const POLL_INTERVAL_MS = 150

/** 실측 FPS 를 평균낼 창 크기. 짧으면 숫자가 튀고 길면 반응이 둔하다. */
const FPS_WINDOW_MS = 2000

export default function ScanPage() {
  const navigate = useNavigate()

  // 'live'    : 카메라를 계속 훑으며 시리얼을 찾는 중
  // 'confirm' : 찾았다. 화면을 멈추고 사용자 확인을 기다린다.
  const [mode, setMode] = useState<'live' | 'confirm'>('live')
  const [result, setResult] = useState<ScanResponse | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [live, setLive] = useState<ScanResponse | null>(null)
  const [scanCount, setScanCount] = useState(0)
  const [fps, setFps] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // 루프가 자기 자신을 중단시킬 수 있도록 최신 mode 를 참조로 들고 있는다.
  const modeRef = useRef(mode)
  modeRef.current = mode

  // 사용자가 "아니요" 로 물린 시리얼. 라벨이 카메라 앞에 그대로 있으면
  // 같은 번호를 즉시 다시 잡아 확인 화면이 무한히 반복되기 때문에 건너뛴다.
  const rejectedRef = useRef<Set<string>>(new Set())
  const [rejectedSerial, setRejectedSerial] = useState<string | null>(null)

  // 스캔이 끝난 시각들. 누적 평균이 아니라 최근 창으로 FPS 를 내야
  // 라벨이나 조명이 바뀌어 OCR 부하가 달라지는 게 숫자에 바로 보인다.
  const fpsSamplesRef = useRef<number[]>([])

  const capture = useCallback((res: ScanResponse) => {
    setResult(res)
    setSelected(res.serial)
    setMode('confirm')
  }, [])

  // --- 실시간 인식 루프 --------------------------------------------------
  // setInterval 이 아니라 순차 실행이다. OCR 은 서버에서 직렬화되므로
  // 요청이 겹치면 큐만 쌓인다.
  useEffect(() => {
    if (mode !== 'live') return

    let cancelled = false
    const controller = new AbortController()

    // 확인 화면에서 돌아왔을 때 멈춰 있던 동안의 표본이 섞이지 않게 비운다.
    fpsSamplesRef.current = []
    setFps(0)

    const loop = async () => {
      while (!cancelled && modeRef.current === 'live') {
        try {
          const res = await scan({ signal: controller.signal, includeFrame: false })
          if (cancelled) return

          setLive(res)
          setScanCount((n) => n + 1)
          setError(null)

          const now = performance.now()
          const samples = fpsSamplesRef.current
          samples.push(now)
          // 창을 벗어난 표본은 버리되, 간격 하나는 남겨 둔다.
          while (samples.length > 2 && now - samples[0] > FPS_WINDOW_MS) samples.shift()
          setFps(samples.length >= 2 ? ((samples.length - 1) * 1000) / (now - samples[0]) : 0)

          if (res.autoCapture && res.serial) {
            if (rejectedRef.current.has(res.serial)) {
              // 물린 번호다. 계속 훑으면서 다른 라벨을 기다린다.
              setRejectedSerial(res.serial)
            } else {
              capture(res)
              return
            }
          } else {
            setRejectedSerial(null)
          }
        } catch (e) {
          if (cancelled || controller.signal.aborted) return
          setError(e instanceof Error ? e.message : String(e))
          // 서버가 죽었을 때 초당 수십 번 때리지 않도록 잠시 쉰다.
          await new Promise((r) => setTimeout(r, 1500))
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS))
      }
    }

    void loop()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [mode, capture])

  // --- 수동 인식 (자동이 안 걸릴 때의 탈출구) ----------------------------
  const manualScan = useCallback(async () => {
    setBusy(true)
    setError(null)
    // 수동 인식은 사용자가 명시적으로 누른 것이므로 거부 목록을 비운다.
    rejectedRef.current.clear()
    setRejectedSerial(null)
    try {
      const res = await scan({ includeFrame: true })
      if (res.serial) {
        capture(res)
      } else {
        setLive(res)
        setError(reasonToMessage(res.reason))
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }, [capture])

  const resume = useCallback(() => {
    // 방금 보여 준 번호를 물린 것으로 기록한다. 그래야 같은 라벨을
    // 비추고 있어도 확인 화면이 다시 뜨지 않는다.
    if (result?.serial) {
      rejectedRef.current.add(result.serial)
      setRejectedSerial(result.serial)
    }
    setResult(null)
    setSelected(null)
    setError(null)
    setMode('live')
  }, [result])

  const { device: registered, loading: lookupLoading } = useDevice(selected ?? undefined)
  const { devices } = useDevices()

  // 스페이스바: 확인 화면에서는 승인, 실시간에서는 수동 인식
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return
      const target = e.target as HTMLElement | null
      if (target && ['INPUT', 'TEXTAREA', 'BUTTON', 'SELECT'].includes(target.tagName)) return
      e.preventDefault()
      if (mode === 'confirm' && selected) {
        navigate(`/result/${encodeURIComponent(selected)}`)
      } else if (!busy) {
        void manualScan()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [mode, selected, busy, manualScan, navigate])

  const hint = live && !live.autoCapture ? autoReasonToMessage(live) : ''

  return (
    <Layout
      step={1}
      title={mode === 'confirm' ? '이 시리얼이 맞습니까?' : '시리얼 번호 인식'}
      subtitle={
        mode === 'confirm'
          ? '자동으로 인식해 화면을 멈췄습니다. 번호를 확인하고 진행하세요.'
          : '기기 라벨을 카메라에 비추면 자동으로 인식합니다. PP-OCRv6 이 DX-M1 NPU 에서 동작합니다.'
      }
    >
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_380px]">
        <CameraStream frozenFrame={mode === 'confirm' ? (result?.frame ?? null) : null} />

        <div className="flex flex-col gap-4">
          {mode === 'live' ? (
            <>
              {/* --- 스캔 중 상태 --- */}
              <div className="dx-card p-5">
                <div className="flex items-center gap-3">
                  <span className="relative flex h-3 w-3">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-dx-cyan opacity-60" />
                    <span className="relative inline-flex h-3 w-3 rounded-full bg-dx-cyan" />
                  </span>
                  <span className="font-mono text-lg font-semibold leading-none tabular-nums">
                    {fps > 0 ? fps.toFixed(1) : '--'}
                  </span>
                  <span className="dx-label">FPS</span>
                  <span className="ml-auto font-mono text-xs text-dx-muted">
                    {scanCount}회 스캔
                  </span>
                </div>

                <p className="mt-3 text-sm text-dx-muted">
                  라벨의 <code className="font-mono text-dx-text">S/N</code>,{' '}
                  <code className="font-mono text-dx-text">SN</code>,{' '}
                  <code className="font-mono text-dx-text">SERIAL</code>,{' '}
                  <code className="font-mono text-dx-text">시리얼</code>,{' '}
                  <code className="font-mono text-dx-text">序列号</code> 표기를 찾고,
                  신뢰도 {live ? (live.autoConfidence * 100).toFixed(0) : 90}% 이상이면
                  자동으로 화면을 멈춥니다.
                </p>

                {live && live.keywordHits.length > 0 && (
                  <p className="mt-2 text-sm text-dx-cyan">
                    표기 감지: {live.keywordHits.join(', ')}
                  </p>
                )}

                {rejectedSerial ? (
                  <p className="mt-2 text-sm text-dx-amber">
                    <span className="font-mono">{rejectedSerial}</span> 는 방금 물린
                    번호라 건너뜁니다. 다른 라벨을 비추거나 아래에서 직접 인식하세요.
                  </p>
                ) : (
                  hint && <p className="mt-2 text-sm text-dx-amber">{hint}</p>
                )}
              </div>

              {live && live.perf && <PerfBadges perf={live.perf} />}

              {live && live.rawTexts.length > 0 && (
                <div className="dx-card p-4">
                  <p className="dx-label mb-2">지금 읽히는 텍스트</p>
                  <ul className="space-y-1">
                    {live.rawTexts.slice(0, 6).map((t, i) => (
                      <li key={`${t}-${i}`} className="truncate font-mono text-sm text-dx-text">
                        {t}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <button
                type="button"
                onClick={() => void manualScan()}
                disabled={busy}
                className="dx-btn-ghost w-full"
              >
                {busy ? '인식 중…' : '지금 바로 인식 (스페이스바)'}
              </button>
            </>
          ) : (
            <>
              {/* --- 확인 --- */}
              <div className="dx-card border-dx-cyan/40 p-5">
                <p className="dx-label mb-1">인식된 시리얼</p>
                <p className="font-mono text-3xl font-bold tracking-wide text-dx-cyan">
                  {selected}
                </p>
                <p className="mt-2 text-sm text-dx-muted">
                  신뢰도{' '}
                  <span className="font-mono text-dx-text">
                    {result ? (result.confidence * 100).toFixed(1) : '—'}%
                  </span>
                  {result && result.keywordHits.length > 0 && (
                    <> · 표기 {result.keywordHits.join(', ')}</>
                  )}
                </p>

                <p className="mt-2 text-sm">
                  {lookupLoading ? (
                    <span className="text-dx-muted">등록 여부 확인 중…</span>
                  ) : registered ? (
                    <span className="text-dx-green">등록된 기기 — {registered.model}</span>
                  ) : (
                    <span className="text-dx-amber">레지스트리에 없는 시리얼입니다.</span>
                  )}
                </p>
              </div>

              <div className="flex flex-col gap-3">
                <button
                  type="button"
                  disabled={!selected}
                  onClick={() => selected && navigate(`/result/${encodeURIComponent(selected)}`)}
                  className="dx-btn-primary w-full py-4 text-lg"
                >
                  맞습니다 · QR 생성 →
                </button>
                <button type="button" onClick={resume} className="dx-btn-ghost w-full">
                  아니요 · 다시 스캔
                </button>
                {!lookupLoading && !registered && selected && (
                  <button
                    type="button"
                    onClick={() =>
                      navigate(`/register?serial=${encodeURIComponent(selected)}`)
                    }
                    className="dx-btn-ghost w-full"
                  >
                    이 기기 등록하기
                  </button>
                )}
              </div>

              {result && result.candidates.length > 1 && (
                <SerialCandidates
                  candidates={result.candidates}
                  selected={selected}
                  onSelect={setSelected}
                />
              )}

              {result && <PerfBadges perf={result.perf} />}

              {result && result.rawTexts.length > 0 && (
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
            <div className="mt-3 flex flex-wrap items-center gap-2">
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
              <button
                type="button"
                onClick={() => navigate('/devices')}
                className="rounded-lg border border-dx-border px-3 py-1.5 text-xs text-dx-muted hover:text-dx-cyan"
              >
                기기 관리 →
              </button>
            </div>
          </details>
        </div>
      </div>
    </Layout>
  )
}
