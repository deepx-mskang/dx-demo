import { useEffect, useRef, useState } from 'react'
import { STREAM_URL } from '../lib/api'

/**
 * C++ 서버가 보내는 MJPEG 스트림을 그대로 <img> 로 받는다.
 * 브라우저 카메라(getUserMedia)를 쓰지 않으므로 HTTPS 가 필요 없다.
 */
export default function CameraStream({ frozenFrame }: { frozenFrame?: string | null }) {
  const imgRef = useRef<HTMLImageElement>(null)
  const [error, setError] = useState(false)
  // 스트림을 재시작할 때 캐시를 우회하기 위한 캐시버스터
  const [nonce, setNonce] = useState(() => Date.now())

  useEffect(() => {
    if (!error) return
    const timer = window.setTimeout(() => {
      setError(false)
      setNonce(Date.now())
    }, 3000)
    return () => window.clearTimeout(timer)
  }, [error])

  return (
    <div className="relative aspect-square w-full overflow-hidden rounded-2xl border border-dx-border bg-black">
      {frozenFrame ? (
        <img
          src={`data:image/jpeg;base64,${frozenFrame}`}
          alt="스캔한 프레임"
          className="h-full w-full object-cover"
        />
      ) : (
        <img
          ref={imgRef}
          src={`${STREAM_URL}?t=${nonce}`}
          alt="카메라 라이브 뷰"
          className="h-full w-full object-cover"
          onError={() => setError(true)}
        />
      )}

      {/* 라벨 정렬 가이드 — OCR 이 보는 영역과 화면이 1:1 로 일치한다 */}
      {!frozenFrame && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="relative h-[28%] w-[78%] rounded-lg border-2 border-dashed border-dx-cyan/70">
            <span className="absolute -top-7 left-0 rounded bg-dx-cyan/90 px-2 py-0.5 text-xs font-semibold text-dx-bg">
              시리얼 라벨을 이 안에 맞추세요
            </span>
          </div>
        </div>
      )}

      {frozenFrame && (
        <span className="absolute left-3 top-3 rounded bg-black/70 px-2 py-1 text-xs font-medium text-dx-text">
          정지 프레임
        </span>
      )}

      {error && !frozenFrame && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-dx-bg/90 p-6 text-center">
          <p className="font-semibold text-dx-red">카메라 스트림에 연결하지 못했습니다.</p>
          <p className="text-sm text-dx-muted">
            서버가 실행 중인지, <code className="font-mono">config.sh</code> 의 카메라 설정이
            맞는지 확인하세요. 3초 후 자동으로 다시 시도합니다.
          </p>
        </div>
      )}
    </div>
  )
}
