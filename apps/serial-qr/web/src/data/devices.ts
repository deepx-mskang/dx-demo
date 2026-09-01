// 기기 정보 타입과 등록 폼 프리필 예시.
//
// 실제 기기 목록은 **서버가 소유**한다 (apps/serial-qr/data/registry.json).
// QR 을 찍은 휴대폰이 같은 데이터를 조회해야 하므로 프론트엔드에 두지 않는다.
// 조회는 lib/api.ts 의 fetchDevices / fetchDevice 를 쓴다.
//
// 최초 실행 시 서버가 data/seed_devices.json 으로 레지스트리를 초기화한다.

export type QaStatus = 'PASS' | 'FAIL' | 'PENDING'

export interface DeviceSpecs {
  tops: number
  memoryGb: number
  powerW: number
}

export interface DeviceInfo {
  serial: string
  model: string
  npu: string
  hwRevision: string
  firmware: string
  manufacturedAt: string
  macAddress: string
  warrantyUntil: string
  qaStatus: QaStatus
  deployedSite: string
  specs: DeviceSpecs
}

export const QA_OPTIONS: { value: QaStatus; label: string }[] = [
  { value: 'PASS', label: 'QA 합격' },
  { value: 'PENDING', label: 'QA 대기' },
  { value: 'FAIL', label: 'QA 불합격' },
]

export const MODEL_OPTIONS = [
  'DEEPX M1 Module',
  'DEEPX M1-Plus Module',
  'DEEPX M1 Dev Board',
]

export const SITE_EXAMPLES = [
  '수원 스마트팩토리 · 검사라인 #3',
  '판교 R&D 센터 · 벤치 #7',
  '평택 물류센터 · 분류기 A',
  '인천 항만 · CCTV 게이트 #12',
  '미배치 (창고 재고)',
]

function isoDate(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

function randomMac(): string {
  // DEEPX OUI 프리픽스는 고정하고 뒤 3바이트만 예시로 만든다.
  const octet = () =>
    Math.floor(Math.random() * 256)
      .toString(16)
      .toUpperCase()
      .padStart(2, '0')
  return `9C:4E:20:${octet()}:${octet()}:${octet()}`
}

/**
 * 등록 폼에 채워 넣을 예시 값을 만든다.
 *
 * - 사전 등록(`/register`)에서는 serial 을 빈 문자열로 넘겨 직접 입력받는다.
 * - 스캔에서 넘어온 경우(`/register?serial=...`)에는 인식된 시리얼을 넣는다.
 *
 * 나머지 필드는 모두 그럴듯한 예시로 미리 채워, 데모 중 타이핑할 일이 없게 한다.
 */
export function makeExampleDraft(serial = ''): DeviceInfo {
  const now = new Date()
  const warranty = new Date(now)
  warranty.setFullYear(warranty.getFullYear() + 3)

  return {
    serial,
    model: MODEL_OPTIONS[0],
    npu: 'DX-M1 (25 TOPS)',
    hwRevision: 'Rev. C2',
    firmware: 'DXRT 2.9.4',
    manufacturedAt: isoDate(now),
    macAddress: randomMac(),
    warrantyUntil: isoDate(warranty),
    qaStatus: 'PASS',
    deployedSite: SITE_EXAMPLES[0],
    specs: { tops: 25, memoryGb: 4, powerW: 5 },
  }
}
