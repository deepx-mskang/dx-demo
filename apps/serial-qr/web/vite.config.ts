import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 개발 중에는 Vite(5173)가 UI 를, C++ 서버(8090)가 카메라/OCR 을 담당한다.
// /api 와 MJPEG 스트림을 C++ 서버로 프록시한다.
const API_TARGET = process.env.DX_SERIAL_QR_API ?? 'http://localhost:8090'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
})
