import { Navigate, Route, Routes } from 'react-router-dom'
import ScanPage from './pages/ScanPage'
import ResultPage from './pages/ResultPage'
import DevicePage from './pages/DevicePage'
import RegisterPage from './pages/RegisterPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ScanPage />} />
      <Route path="/result/:serial" element={<ResultPage />} />
      <Route path="/device/:serial" element={<DevicePage />} />
      <Route path="/register" element={<RegisterPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
