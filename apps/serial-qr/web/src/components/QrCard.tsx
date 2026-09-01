import { useEffect, useRef, useState } from 'react'
import QRCode from 'qrcode'

export default function QrCard({ url }: { url: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    QRCode.toCanvas(canvas, url, {
      width: 320,
      margin: 2,
      errorCorrectionLevel: 'M',
      // 휴대폰 카메라 인식률을 위해 QR 자체는 항상 밝은 배경으로 그린다.
      color: { dark: '#0a0e14', light: '#ffffff' },
    }).catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)))
  }, [url])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(url)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      setError('클립보드에 복사하지 못했습니다.')
    }
  }

  return (
    <div className="dx-card flex flex-col items-center gap-4 p-6">
      <div className="rounded-xl bg-white p-3">
        <canvas ref={canvasRef} className="block" />
      </div>

      <div className="w-full text-center">
        <p className="dx-label mb-1">QR 인코딩 주소</p>
        {/* 휴대폰이 없거나 QR 인식이 안 될 때를 위해 URL 을 그대로 노출한다 */}
        <p className="break-all font-mono text-sm text-dx-text">{url}</p>
      </div>

      <button type="button" onClick={copy} className="dx-btn-ghost w-full py-2 text-sm">
        {copied ? '복사됨 ✓' : 'URL 복사'}
      </button>

      <p className="text-center text-xs text-dx-muted">
        휴대폰 기본 카메라로 QR 을 비추면 기기 정보 페이지가 열립니다.
        <br />
        휴대폰이 같은 네트워크에 있어야 합니다.
      </p>

      {error && <p className="text-sm text-dx-red">{error}</p>}
    </div>
  )
}
