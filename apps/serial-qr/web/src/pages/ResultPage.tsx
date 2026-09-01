import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import DeviceInfoCard from '../components/DeviceInfoCard'
import Layout from '../components/Layout'
import QrCard from '../components/QrCard'
import { buildDeviceUrl, fetchConfig, type ServerConfig } from '../lib/api'
import { useDevice } from '../hooks/useDevice'

export default function ResultPage() {
  const { serial = '' } = useParams()
  const navigate = useNavigate()
  const [config, setConfig] = useState<ServerConfig | null>(null)
  const [configError, setConfigError] = useState(false)
  const { device, loading } = useDevice(serial)

  useEffect(() => {
    fetchConfig()
      .then(setConfig)
      .catch(() => setConfigError(true))
  }, [])

  const url = buildDeviceUrl(serial, config)

  return (
    <Layout
      step={2}
      title="QR 코드 발행"
      subtitle="이 QR 을 휴대폰 카메라로 찍으면 기기 정보 조회 페이지가 열립니다."
    >
      <div className="grid gap-6 lg:grid-cols-[380px_minmax(0,1fr)]">
        <div className="space-y-4">
          <QrCard url={url} />

          {configError && (
            <p className="rounded-xl border border-dx-amber/40 bg-dx-amber/10 p-3 text-xs text-dx-amber">
              서버에서 LAN 주소를 가져오지 못해 현재 브라우저 주소를 사용했습니다.
              휴대폰에서 열리지 않으면 서버 로그의 LAN base URL 을 확인하세요.
            </p>
          )}

          <div className="flex gap-3">
            <button type="button" onClick={() => navigate('/')} className="dx-btn-ghost flex-1">
              ← 다시 스캔
            </button>
            <Link to={`/device/${encodeURIComponent(serial)}`} className="dx-btn-primary flex-1">
              조회 화면 열기
            </Link>
          </div>
        </div>

        <div>
          {loading && <div className="dx-card h-80 animate-pulse" />}

          {!loading && device && <DeviceInfoCard device={device} />}

          {!loading && !device && (
            <div className="dx-card p-6">
              <p className="dx-label">시리얼 번호</p>
              <p className="font-mono text-xl font-bold text-dx-cyan">{serial}</p>
              <p className="mt-4 text-sm text-dx-amber">
                레지스트리에 등록되지 않은 시리얼입니다. QR 은 정상적으로 생성되지만,
                조회 화면에서는 미등록 기기로 표시됩니다.
              </p>
              <Link
                to={`/register?serial=${encodeURIComponent(serial)}`}
                className="dx-btn-primary mt-5 w-full"
              >
                지금 등록하기
              </Link>
            </div>
          )}
        </div>
      </div>
    </Layout>
  )
}
