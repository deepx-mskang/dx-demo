import { useEffect, useRef, useState } from 'react'
import QRCode from 'qrcode'

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c] ?? c,
  )

export default function QrCard({ url, serial }: { url: string; serial?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [error, setError] = useState<string | null>(null)

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

  // 라벨 프린터로 바로 뽑을 수 있도록 숨은 iframe 에 인쇄용 문서를 만들어 출력한다.
  // (window.open 은 팝업 차단에 걸리므로 iframe 을 쓴다)
  const print = () => {
    const canvas = canvasRef.current
    if (!canvas) return

    const frame = document.createElement('iframe')
    frame.setAttribute('aria-hidden', 'true')
    frame.style.cssText = 'position:fixed;right:0;bottom:0;width:0;height:0;border:0'
    document.body.appendChild(frame)

    const doc = frame.contentDocument
    const win = frame.contentWindow
    if (!doc || !win) {
      frame.remove()
      setError('프린트 창을 열지 못했습니다.')
      return
    }

    doc.open()
    doc.write(`<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>${escapeHtml(serial || 'QR')}</title>
<style>
  @page { margin: 10mm; }
  body { margin:0; display:flex; flex-direction:column; align-items:center; gap:4mm;
         font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color:#000; }
  img { width:55mm; height:55mm; image-rendering:pixelated; }
  .serial { font-size:16pt; font-weight:700; letter-spacing:0.04em; }
  .url { font-size:7pt; max-width:70mm; text-align:center; word-break:break-all; color:#333; }
</style></head><body>
<img src="${canvas.toDataURL('image/png')}" alt="QR">
${serial ? `<div class="serial">${escapeHtml(serial)}</div>` : ''}
<div class="url">${escapeHtml(url)}</div>
</body></html>`)
    doc.close()

    const run = () => {
      win.focus()
      win.print()
      window.setTimeout(() => frame.remove(), 1000)
    }

    const img = doc.querySelector('img')
    if (img && !img.complete) img.addEventListener('load', run, { once: true })
    else run()
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

      <button type="button" onClick={print} className="dx-btn-ghost w-full py-2 text-sm">
        QR 프린트
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
