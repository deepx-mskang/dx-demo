// C++ 서버(serial_ocr_server) API 클라이언트

import type { DeviceInfo } from '../data/devices'

export interface SerialCandidate {
  text: string
  rawText: string
  score: number
  normalized: boolean
}

export interface OcrPerf {
  detMs: number
  recMs: number
  e2eMs: number
  numBoxes: number
  numCrops: number
  totalChars: number
  cps: number
}

export interface ScanResponse {
  ok: boolean
  reason?: string
  serial: string | null
  confidence: number
  candidates: SerialCandidate[]
  rawTexts: string[]
  perf: OcrPerf
  /** 스캔 시점 프레임 (base64 JPEG, data: 접두사 없음) */
  frame?: string
}

export interface ServerConfig {
  lanBaseUrl: string
  port: number
}

export const STREAM_URL = '/api/stream'

export async function scan(signal?: AbortSignal): Promise<ScanResponse> {
  const res = await fetch('/api/scan', { method: 'POST', signal })
  const body = await res.json().catch(() => null)

  if (!res.ok) {
    const reason = body?.reason ?? `http_${res.status}`
    throw new Error(reasonToMessage(reason))
  }
  if (!body) {
    throw new Error('서버 응답을 해석하지 못했습니다.')
  }
  return body as ScanResponse
}

export async function fetchConfig(): Promise<ServerConfig> {
  const res = await fetch('/api/config')
  if (!res.ok) {
    throw new Error('서버 설정을 가져오지 못했습니다.')
  }
  return (await res.json()) as ServerConfig
}

export function reasonToMessage(reason: string | undefined): string {
  switch (reason) {
    case 'no_serial_found':
      return '시리얼 번호를 찾지 못했습니다. 라벨을 가이드 안에 맞추고 다시 시도하세요.'
    case 'camera_unavailable':
      return '카메라를 열 수 없습니다. config.sh 의 DX_CAMERA_IDX 를 확인하세요.'
    case 'no_frame':
      return '아직 카메라 프레임이 준비되지 않았습니다. 잠시 후 다시 시도하세요.'
    case 'ocr_error':
      return 'OCR 추론 중 오류가 발생했습니다. 서버 로그를 확인하세요.'
    default:
      return reason ? `요청에 실패했습니다 (${reason}).` : '요청에 실패했습니다.'
  }
}

/**
 * QR 에 넣을 조회 URL을 만든다.
 *
 * 서버가 알려준 LAN 주소를 우선 쓴다. 데모 PC 에서 localhost 로 열어 놓고
 * window.location.origin 을 쓰면 휴대폰이 그 QR 을 열 수 없기 때문이다.
 */
export function buildDeviceUrl(serial: string, config: ServerConfig | null): string {
  const base = config?.lanBaseUrl?.trim() || window.location.origin
  return `${base.replace(/\/$/, '')}/device/${encodeURIComponent(serial)}`
}

// ---------------------------------------------------------------------------
// 기기 레지스트리
//
// 기기 정보는 서버가 소유한다(data/registry.json). QR 을 찍은 휴대폰이 같은
// 데이터를 봐야 하므로 브라우저 저장소에 두지 않는다. 시리얼은 유니크하다.
// ---------------------------------------------------------------------------

export interface RegisterOutcome {
  ok: boolean
  reason?: 'duplicate_serial' | 'invalid_serial' | 'invalid_payload' | 'write_failed' | 'bad_json'
  message: string
  device?: DeviceInfo
}

export async function fetchDevices(): Promise<DeviceInfo[]> {
  const res = await fetch('/api/devices')
  if (!res.ok) {
    throw new Error('기기 목록을 가져오지 못했습니다.')
  }
  return (await res.json()) as DeviceInfo[]
}

/** 등록되지 않은 시리얼이면 null. */
export async function fetchDevice(serial: string): Promise<DeviceInfo | null> {
  const res = await fetch(`/api/devices/${encodeURIComponent(serial)}`)
  if (res.status === 404) {
    return null
  }
  if (!res.ok) {
    throw new Error('기기 정보를 가져오지 못했습니다.')
  }
  return (await res.json()) as DeviceInfo
}

export async function registerDevice(device: DeviceInfo): Promise<RegisterOutcome> {
  const res = await fetch('/api/devices', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(device),
  })
  const body = (await res.json().catch(() => null)) as RegisterOutcome | null
  if (!body) {
    return { ok: false, message: `등록에 실패했습니다 (HTTP ${res.status}).` }
  }
  return body
}

export async function deleteDevice(serial: string): Promise<boolean> {
  const res = await fetch(`/api/devices/${encodeURIComponent(serial)}`, { method: 'DELETE' })
  return res.ok
}
