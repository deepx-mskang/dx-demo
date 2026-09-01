import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import Layout from '../components/Layout'
import { deleteDevice } from '../lib/api'
import { useDevices } from '../hooks/useDevice'
import type { DeviceInfo, QaStatus } from '../data/devices'

const QA_STYLE: Record<QaStatus, string> = {
  PASS: 'bg-dx-green/15 text-dx-green border-dx-green/30',
  PENDING: 'bg-dx-amber/15 text-dx-amber border-dx-amber/30',
  FAIL: 'bg-dx-red/15 text-dx-red border-dx-red/30',
}

const QA_LABEL: Record<QaStatus, string> = {
  PASS: '합격',
  PENDING: '대기',
  FAIL: '불합격',
}

/** 등록된 기기 목록과 삭제. 데모 진행자가 쓰는 화면이다. */
export default function DevicesPage() {
  const navigate = useNavigate()
  const { devices, loading, reload } = useDevices()

  // 삭제는 되돌릴 수 없으므로 같은 자리에서 한 번 더 확인받는다.
  const [pending, setPending] = useState<string | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [removed, setRemoved] = useState<string | null>(null)

  const remove = async (serial: string) => {
    setBusy(serial)
    setError(null)
    try {
      if (await deleteDevice(serial)) {
        setRemoved(serial)
        reload()
      } else {
        setError(`${serial} 삭제에 실패했습니다. 이미 지워졌을 수 있습니다.`)
        reload()
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(null)
      setPending(null)
    }
  }

  return (
    <Layout
      title="등록 기기 관리"
      subtitle="서버 레지스트리에 등록된 기기입니다. 삭제하면 QR 로 조회해도 미등록으로 표시됩니다."
    >
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-sm text-dx-muted">
            총{' '}
            <span className="font-mono text-lg font-bold text-dx-cyan">
              {loading ? '—' : devices.length}
            </span>
            대
          </p>
          <div className="ml-auto flex gap-3">
            <button type="button" onClick={() => navigate('/')} className="dx-btn-ghost px-4 py-2 text-sm">
              ← 스캔으로
            </button>
            <Link to="/register" className="dx-btn-primary px-4 py-2 text-sm">
              + 사전 등록
            </Link>
          </div>
        </div>

        {removed && (
          <div className="rounded-xl border border-dx-border bg-dx-surface p-3 text-sm text-dx-muted">
            <span className="font-mono text-dx-text">{removed}</span> 를 삭제했습니다.
          </div>
        )}

        {error && (
          <div className="rounded-xl border border-dx-red/40 bg-dx-red/10 p-4 text-sm text-dx-red">
            {error}
          </div>
        )}

        {loading && <div className="dx-card h-64 animate-pulse" />}

        {!loading && devices.length === 0 && (
          <div className="dx-card p-10 text-center">
            <p className="text-dx-muted">등록된 기기가 없습니다.</p>
            <Link to="/register" className="dx-btn-primary mx-auto mt-4 w-fit px-6">
              기기 사전 등록
            </Link>
          </div>
        )}

        {!loading &&
          devices.map((d: DeviceInfo) => {
            const confirming = pending === d.serial
            return (
              <div
                key={d.serial}
                className={[
                  'dx-card flex flex-wrap items-center gap-x-4 gap-y-3 p-4 transition-colors',
                  confirming ? 'border-dx-red/60' : '',
                ].join(' ')}
              >
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-lg font-bold text-dx-cyan">{d.serial}</span>
                    <span
                      className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${QA_STYLE[d.qaStatus]}`}
                    >
                      {QA_LABEL[d.qaStatus]}
                    </span>
                  </div>
                  <p className="mt-1 truncate text-sm text-dx-muted">
                    {d.model} · {d.deployedSite}
                  </p>
                </div>

                {confirming ? (
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-dx-red">삭제할까요?</span>
                    <button
                      type="button"
                      disabled={busy === d.serial}
                      onClick={() => void remove(d.serial)}
                      className="dx-btn rounded-xl bg-dx-red px-4 py-2 text-sm text-white hover:opacity-90"
                    >
                      {busy === d.serial ? '삭제 중…' : '삭제'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setPending(null)}
                      className="dx-btn-ghost px-4 py-2 text-sm"
                    >
                      취소
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Link
                      to={`/device/${encodeURIComponent(d.serial)}`}
                      className="dx-btn-ghost px-4 py-2 text-sm"
                    >
                      조회
                    </Link>
                    <Link
                      to={`/result/${encodeURIComponent(d.serial)}`}
                      className="dx-btn-ghost px-4 py-2 text-sm"
                    >
                      QR
                    </Link>
                    <button
                      type="button"
                      onClick={() => {
                        setPending(d.serial)
                        setRemoved(null)
                      }}
                      className="dx-btn-ghost px-4 py-2 text-sm text-dx-red hover:bg-dx-red/10"
                    >
                      삭제
                    </button>
                  </div>
                )}
              </div>
            )
          })}

        <p className="pt-2 text-xs text-dx-muted">
          기본 8대를 포함해 모두 되돌리려면 서버를 멈추고{' '}
          <code className="font-mono">apps/serial-qr/data/registry.json</code> 을 지운 뒤 다시
          실행하세요. 시드로 재생성됩니다.
        </p>
      </div>
    </Layout>
  )
}
