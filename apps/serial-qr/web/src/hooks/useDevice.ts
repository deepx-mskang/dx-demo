import { useCallback, useEffect, useState } from 'react'
import { fetchDevice, fetchDevices } from '../lib/api'
import type { DeviceInfo } from '../data/devices'

interface DeviceState {
  device: DeviceInfo | null
  loading: boolean
  error: string | null
  reload: () => void
}

/** 시리얼 하나를 서버 레지스트리에서 조회한다. 미등록이면 device 가 null. */
export function useDevice(serial: string | undefined): DeviceState {
  const [device, setDevice] = useState<DeviceInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    if (!serial) {
      setDevice(null)
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    fetchDevice(serial)
      .then((d) => {
        if (!cancelled) setDevice(d)
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [serial, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { device, loading, error, reload }
}

/** 등록된 기기 전체 목록. */
export function useDevices(): { devices: DeviceInfo[]; loading: boolean; reload: () => void } {
  const [devices, setDevices] = useState<DeviceInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetchDevices()
      .then((d) => {
        if (!cancelled) setDevices(d)
      })
      .catch(() => {
        if (!cancelled) setDevices([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  return { devices, loading, reload }
}
