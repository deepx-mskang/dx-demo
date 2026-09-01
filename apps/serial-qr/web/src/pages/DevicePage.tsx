import { Link, useParams } from 'react-router-dom'
import DeviceInfoCard from '../components/DeviceInfoCard'
import Layout from '../components/Layout'
import { useDevice } from '../hooks/useDevice'

/**
 * QR 을 찍은 휴대폰이 도착하는 화면. 모바일 우선 레이아웃이다.
 * 기기 정보는 서버 레지스트리에서 조회하므로 등록한 PC 가 아니어도 보인다.
 */
export default function DevicePage() {
  const { serial = '' } = useParams()
  const { device, loading, error } = useDevice(serial)

  const title = loading ? '조회 중…' : device ? '기기 정보' : '미등록 기기'
  const subtitle = loading
    ? '서버에서 기기 정보를 가져오는 중입니다.'
    : device
      ? 'QR 로 조회된 기기의 상세 정보입니다.'
      : 'QR 에 담긴 시리얼이 레지스트리에 없습니다.'

  return (
    <Layout step={3} title={title} subtitle={subtitle}>
      <div className="mx-auto max-w-xl space-y-4">
        {loading && <div className="dx-card h-64 animate-pulse" />}

        {!loading && error && (
          <div className="rounded-xl border border-dx-red/40 bg-dx-red/10 p-4 text-sm text-dx-red">
            {error}
          </div>
        )}

        {!loading && !error && device && <DeviceInfoCard device={device} />}

        {!loading && !error && !device && (
          <div className="dx-card p-6 text-center">
            <p className="dx-label">조회한 시리얼</p>
            <p className="mt-1 break-all font-mono text-xl font-bold text-dx-cyan">{serial}</p>
            <p className="mt-4 text-sm text-dx-muted">
              이 시리얼로 등록된 기기를 찾을 수 없습니다. 시리얼이 잘못 인식되었거나
              아직 등록되지 않은 기기입니다.
            </p>
            <Link
              to={`/register?serial=${encodeURIComponent(serial)}`}
              className="dx-btn-primary mt-5 w-full"
            >
              이 기기 등록하기
            </Link>
          </div>
        )}

        <Link to="/" className="dx-btn-ghost w-full">
          새 기기 스캔하기
        </Link>
      </div>
    </Layout>
  )
}
