/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // 기존 DEEPX 데모(perf-monitor)의 다크/시안 톤에 맞춘 팔레트
        dx: {
          bg: '#0a0e14',
          surface: '#131923',
          card: '#1a2130',
          border: '#273044',
          cyan: '#00d4e0',
          cyanDim: '#0891a0',
          text: '#e6edf5',
          muted: '#8494ad',
          green: '#2ecc8f',
          amber: '#f5a524',
          red: '#f2555a',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
}
