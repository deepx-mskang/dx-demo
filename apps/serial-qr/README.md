# Serial-QR — 시리얼 OCR → QR 발행 → 기기 조회 웹 데모

카메라로 기기 라벨의 **시리얼 번호를 읽고**, 그 시리얼로 **QR 코드를 발행**하고,
그 QR 을 휴대폰으로 찍으면 **기기 정보 페이지가 열리는** 3단계 웹 데모입니다.

OCR 은 `apps/paddle-ocr` 의 PP-OCRv6 엔진(DX-M1 NPU)을 그대로 재사용합니다.

```
[카메라] --MJPEG--> [브라우저]
                        |  "시리얼 인식"
                        v
   POST /api/scan --> [PP-OCRv6 @ DX-M1 NPU] --> DX-M1-A7K3P9V2
                        |
                        v
                   [QR 발행]  http://<LAN-IP>:8090/device/DX-M1-A7K3P9V2
                        |
                   휴대폰으로 촬영
                        v
                   [기기 정보 조회]
```

## 구성

| 구성 요소 | 경로 | 설명 |
|---|---|---|
| C++ 서버 | `cpp/serial_ocr_server.cpp` | 카메라 캡처(V4L2) + MJPEG 스트리밍 + OCR + 정적 파일 서빙 |
| OCR 엔진 | `../paddle-ocr/cpp/ocr_engine.cpp` | **재사용**. 이 앱은 소스를 참조만 하며 수정하지 않습니다 |
| 웹 UI | `web/` | Vite + React + TypeScript + Tailwind |
| 기기 관리 화면 | `web/src/pages/DevicesPage.tsx` | 등록 목록 조회·삭제 (`/devices`) |
| 기기 레지스트리 | `data/registry.json` | **서버 소유**. 등록된 기기 정보. 시리얼은 유니크 |
| 시드 데이터 | `data/seed_devices.json` | 최초 실행 시 레지스트리를 초기화하는 기본 8대 |
| 라벨 생성기 | `tools/make_labels.py` | 데모용 시리얼 라벨 시트 PNG 생성 |

브라우저 카메라(`getUserMedia`)를 쓰지 않고 서버가 카메라를 잡아 MJPEG 으로 흘려보내므로
**HTTPS 설정이 필요 없습니다.**

## 사전 준비

### Node.js 20 이상

프론트엔드가 Vite + React 라 Node 가 필요합니다. 이 레포에서 Node 를 쓰는 앱은 이 데모가 처음입니다.

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
source ~/.nvm/nvm.sh && nvm install 20
```

### 모델 자산

`workspace/models/ocr/v6/` 의 PP-OCRv6 모델을 사용합니다. 없으면 실행 스크립트가 자동으로
`setup_assets.sh` 를 호출합니다.

## 빌드

```bash
apps/serial-qr/build.sh            # C++ 서버 + 웹 프론트엔드
apps/serial-qr/build.sh --clean    # 전체 재빌드
apps/serial-qr/build.sh --no-web   # C++ 서버만 (Node 없이)
```

## 실행

```bash
scripts/run_serial_qr.sh     # 서버 기동 후 브라우저 자동 실행
scripts/kill_serial_qr.sh    # 종료
```

기본 포트는 **8090** 입니다 (`config.sh` 의 `DX_SERIAL_QR_PORT`).
카메라는 `config.sh` 의 `DX_CAMERA_DEV` / `DX_CAMERA_IDX` 를 따릅니다.

> ⚠️ 이 데모와 `run_ocr.sh` 는 같은 NPU 와 같은 카메라를 씁니다. 동시에 실행할 수 없어
> `run_serial_qr.sh` 가 시작할 때 `kill_ocr.sh` 를 먼저 호출합니다.

## 데모 진행 방법

1. **라벨 준비** — `assets/serial_labels.png` 를 인쇄하거나 휴대폰 화면에 띄웁니다.
2. **비추기** — 라벨을 가이드 안에 넣기만 하면 됩니다. **버튼을 누를 필요가 없습니다.**
   자동으로 인식되면 화면이 멈추고 번호를 보여 줍니다.
3. **확인** — `맞습니다 · QR 생성 →` (스페이스바). 틀렸으면 `아니요 · 다시 스캔`.
4. **조회** — 휴대폰 기본 카메라로 QR 을 비추면 기기 정보 페이지가 열립니다.
   **휴대폰이 데모 PC 와 같은 네트워크에 있어야 합니다.**

라벨이 없을 때는 화면 하단의 `라벨 없이 시연하기` 에서 시리얼을 직접 골라
3~4 단계만 시연할 수 있습니다.

## 실시간 자동 인식

스캔 화면은 버튼을 기다리지 않고 **카메라를 계속 훑습니다**. 프레임마다 OCR 을 돌리고,
아래 조건을 만족하면 화면을 멈춘 뒤 사용자 확인을 받습니다.

**조건 1 — 신뢰도** 인식 신뢰도가 임계값 이상 (기본 **90%**, `--auto-confidence` 로 조정)

**조건 2 — 근거** 다음 중 하나

| 근거 (`autoReason`) | 설명 |
|---|---|
| `keyword_same_box` | 시리얼과 **같은 줄**에 시리얼 표기가 있음 — 가장 강함 |
| `keyword` | 프레임 어딘가에 시리얼 표기가 있음 |
| `strict_format` | 표기는 없지만 정식 포맷(`DX-M1-XXXXXXXX`)으로 읽힘 |

인식하는 시리얼 표기:

| 언어 | 표기 |
|---|---|
| 영문 | `S/N`, `S.N`, `SN`, `SERIAL`, `SERIAL NO` |
| 한국어 | `시리얼`, `일련번호`, `제품번호` |
| 중국어 | `序列号`, `序列號`, `序號`, `編號` |
| 일본어 | `シリアル`, `製造番号` |

`SN` 처럼 짧은 표기는 **단어 경계**를 확인해 다른 단어에 묻힌 경우를 걸러냅니다.
표기 검사는 OCR **원문**에서 합니다 — 정규화가 `S/N` 을 `S-N` 으로 바꾸기 때문입니다.

### 동작 세부

- **폴링 방식** — 프론트엔드가 `/api/scan` 을 순차 호출합니다(`setInterval` 아님).
  OCR 이 서버에서 직렬화되므로 요청이 겹치면 큐만 쌓입니다.
- **대역폭** — 실시간 폴링은 `?frame=0` 으로 호출해 base64 JPEG(~80KB)를 받지 않습니다.
  서버는 **자동 캡처가 걸린 프레임에만** 이미지를 붙여 줍니다. 그래서 확인 화면에 뜨는
  정지 화면은 정확히 인식이 일어난 그 프레임입니다.
- **거부한 번호는 건너뜁니다** — `아니요 · 다시 스캔` 을 누르면 그 시리얼을 기억합니다.
  라벨을 카메라 앞에 그대로 두고 있어도 확인 화면이 무한 반복되지 않습니다.
  `지금 바로 인식` 을 누르면 기억을 지우고 다시 잡습니다.
- **수동 인식** — 자동이 안 걸릴 때를 위해 `지금 바로 인식 (스페이스바)` 버튼이 있습니다.
  신뢰도·표기 조건을 무시하고 잡힌 것을 그대로 보여 줍니다.

## 기기 등록

기기 정보는 **서버가 소유**한다 (`data/registry.json`). 브라우저 저장소가 아니라 서버에
두는 이유는, QR 을 찍은 **휴대폰이 같은 데이터를 조회**해야 하기 때문이다.
등록 즉시 반영되며 재빌드가 필요 없다.

**시리얼은 유니크하다.** 이미 등록된 시리얼로 등록하면 거부된다(HTTP 409). 대소문자와
앞뒤 공백은 정규화하므로 `dx-m1-a7k3p9v2` 도 같은 기기로 취급한다.

### 등록 경로 두 가지

| 경로 | 진입 | 시리얼 칸 | 나머지 항목 |
|---|---|---|---|
| **사전 등록** | 스캔 화면의 `+ 기기 사전 등록` → `/register` | **비어 있음** (커서 자동 포커스) | 예시로 프리필 |
| **인식 후 등록** | 미등록 시리얼 스캔 → `이 기기 등록하기` → `/register?serial=…` | **인식된 시리얼 자동 입력** (읽기 전용) | 예시로 프리필 |

두 경로 모두 모델·펌웨어·MAC·제조일·보증 만료·QA·배치 위치가 그럴듯한 값으로 미리
채워져 있어, 데모 중에는 시리얼만 넣고 바로 등록하면 된다. 등록이 끝나면 곧바로
QR 발행 화면으로 넘어간다.

미등록 시리얼은 조회 화면(`/device/…`)에서도 `이 기기 등록하기` 로 이어진다.

### 기기 삭제

헤더의 **`기기 관리`** (`/devices`) 에서 등록된 기기를 보고 삭제할 수 있습니다.
스캔 화면의 `라벨 없이 시연하기` 안에도 진입 버튼이 있습니다.

삭제는 되돌릴 수 없으므로 **같은 자리에서 한 번 더 확인**을 받습니다
(`삭제` → `삭제할까요?` → `삭제` / `취소`). 삭제하면 그 시리얼은 QR 로 조회해도
미등록으로 표시됩니다.

조회 화면(`/device/…`)에는 삭제 버튼을 두지 않았습니다. QR 을 찍은 **휴대폰이 여는
화면**이라, 조회하러 온 사람이 실수로 지울 수 있기 때문입니다.

CLI 로도 됩니다:

```bash
curl -X DELETE localhost:8090/api/devices/DX-M1-A7K3P9V2
```

### 시드와 초기화

레지스트리 파일이 없으면 서버가 `data/seed_devices.json` 으로 자동 생성한다.
데모를 처음 상태로 되돌리려면:

```bash
scripts/kill_serial_qr.sh
rm apps/serial-qr/data/registry.json
scripts/run_serial_qr.sh
```

기본 8대는 `data/seed_devices.json` 에서 편집한다. 인쇄용 라벨 시트
(`assets/serial_labels.png`, A4 300dpi 8칸)는 아래로 다시 만든다:

```bash
source .venv/bin/activate && python3 apps/serial-qr/tools/make_labels.py
```

> 라벨 생성기는 시드 JSON 을 읽는다. 시리얼을 바꿨다면 라벨도 다시 뽑을 것.

## 시리얼 판정 기준

`cpp/serial_ocr_server.cpp` 의 `extractSerials()` 가 OCR 텍스트에서 시리얼을 뽑는 규칙입니다.

**정규화** — 영숫자는 대문자로 남기고, 나머지 문자(공백 `:` `/` `.` `_` …)는 `-` 하나로 접습니다.
지우지 않고 `-` 로 바꾸는 이유는 토큰 경계를 살리기 위해서입니다
(`S/N: DX-M1-A7K3P9V2` → `SN-DX-M1-A7K3P9V2`). 덕분에 OCR 이 하이픈을 공백으로 읽은
`DX M1 A7K3P9V2` 도 같이 복구됩니다.

**탐색** — 전체 일치가 아니라 **부분 탐색**입니다. 시리얼 앞뒤에 다른 문구가 같은 텍스트
박스로 묶여 나와도 찾아냅니다. 우선순위 순으로:

| 순위 | 패턴 | 조건 |
|---|---|---|
| 0 | `DX-?M1-?([A-Z0-9]{8})` | 정식 포맷, 본문 정확히 8자 |
| 1 | `([A-Z]{2,4})-?([A-Z0-9]{6,12})` | 일반 시리얼 형태 + 본문에 숫자 3개 이상 |
| 2 | 0번 패턴을 **모든 박스를 이어 붙인 문자열**에 재적용 | 시리얼이 여러 박스로 쪼개진 경우 |
| 3 | `DX-?M1-?([A-Z0-9]{5,12})` + 숫자 2개 이상 | 자릿수가 어긋난 경우. 0~2 에서 못 찾았을 때만 |

매칭은 영숫자 한가운데서 시작하거나 끝나지 않아야 합니다(`XDX-M1-...` 같은 오탐 방지).
숫자 개수 조건은 라벨의 설명 문구가 시리얼로 오인되는 것을 막습니다
(`S/N: DX-M1 NPU MODULE` 은 어떤 순위에도 걸리지 않습니다).

시리얼을 못 찾으면 UI 의 **`OCR 원본 텍스트`** 를 펼쳐 무엇이 읽혔는지 확인하세요.
거기에 시리얼이 보이는데도 못 찾는다면 위 규칙 중 어디서 걸리는지 알 수 있습니다.

> **시리얼 작명 규칙**: 본문 8자리에 `O Q I L S B Z` 를 쓰지 마세요.
> 서버가 OCR 혼동 문자를 숫자로 보정하기 때문에(`O→0`, `I→1`, `S→5` …),
> 이 글자들이 정답에 들어 있으면 정상 인식된 값이 오히려 훼손됩니다.

## API

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/api/health` | `{"status":"ok","camera":true,"npu":true,"frames":N}` |
| `GET` | `/api/config` | `{"lanBaseUrl":"http://192.168.x.x:8090","port":8090}` |
| `GET` | `/api/stream` | MJPEG (`multipart/x-mixed-replace`) |
| `POST` | `/api/scan` | 최신 프레임 OCR → 시리얼 추출. `?frame=0` 이면 자동 캡처 시에만 이미지 포함 |
| `GET` | `/api/devices` | 등록된 기기 전체 |
| `GET` | `/api/devices/{serial}` | 단건 조회 (미등록이면 404) |
| `POST` | `/api/devices` | 기기 등록 (201 / 중복 409 / 형식 오류 400) |
| `DELETE` | `/api/devices/{serial}` | 등록 취소 (데모 반복 시연용) |
| `GET` | `/*` | `web/dist` 정적 서빙, 404 는 `index.html` 로 폴백(SPA 라우팅) |

`/api/scan` 응답:

```json
{
  "ok": true,
  "serial": "DX-M1-A7K3P9V2",
  "confidence": 0.9997,
  "candidates": [
    { "text": "DX-M1-A7K3P9V2", "rawText": "DX-M1-A7K3P9V2",
      "score": 0.9997, "normalized": false }
  ],
  "rawTexts": ["DEEPX", "DX-M1-A7K3P9V2", "S/N:DX-M1 NPU MODULE"],
  "perf": { "detMs": 155.3, "recMs": 12.4, "e2eMs": 196.3, "numBoxes": 3,
            "numCrops": 3, "totalChars": 39, "cps": 198.7 },
  "autoCapture": true,
  "autoReason": "keyword_same_box",
  "autoConfidence": 0.90,
  "keywordHits": ["S/N"],
  "frame": "<base64 jpeg>"
}
```

**QR 주소는 `/api/config` 의 `lanBaseUrl` 을 씁니다.** 프론트가 `window.location.origin`
을 쓰면 데모 PC 에서 `localhost` 로 열었을 때 만들어진 QR 을 휴대폰이 열 수 없기 때문에,
서버가 `getifaddrs()` 로 찾은 LAN 주소를 직접 알려줍니다.

## 개발

```bash
# 터미널 1 — C++ 서버 (카메라 + OCR)
./apps/serial-qr/cpp/build/serial_ocr_server --device /dev/video0 --port 8090

# 터미널 2 — Vite 개발 서버 (:5173, /api 는 8090 으로 프록시)
cd apps/serial-qr/web && npm run dev
```

### 카메라 없이 OCR 만 검증

```bash
./apps/serial-qr/cpp/build/serial_ocr_server --test-image apps/serial-qr/assets/serial_labels.png
```

이미지 1장으로 OCR 을 돌리고 `/api/scan` 과 동일한 JSON 을 출력한 뒤 종료합니다.
시리얼 정규식이나 혼동 문자 보정 규칙을 손볼 때 사용하세요.

전체 옵션은 `--help` 를 참고하세요.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 카메라 화면이 안 나옴 | `ls /dev/video*` 로 실제 장치 확인 후 `config.sh` 의 `DX_CAMERA_IDX` 수정. 다른 데모가 카메라를 잡고 있으면 `scripts/kill_all.sh` |
| 휴대폰에서 QR 이 안 열림 | ① 휴대폰이 같은 네트워크인지 ② 서버 로그의 `LAN base URL` 이 `localhost` 가 아닌지 ③ 방화벽이 8090 을 막는지. QR 아래 URL 을 직접 입력해도 됩니다 |
| 자동으로 안 멈춤 | 화면의 안내 문구를 보세요. `신뢰도 87% — 임계값 90% 미만` 이면 라벨을 더 가까이. 안 되면 `지금 바로 인식` 버튼 |
| 같은 번호가 계속 뜸 | `아니요` 를 누르면 그 번호는 건너뜁니다. 라벨을 치우거나 다른 라벨을 비추세요 |
| 시리얼을 못 읽음 | 라벨을 가이드 안에 크게, 정면으로. `지금 읽히는 텍스트` 로 무엇이 읽혔는지 확인 |
| 엉뚱한 문자로 읽힘 | 시리얼에 `O Q I L S B Z` 가 들어갔는지 확인 (위 작명 규칙 참고) |
| `npm not found` | `source ~/.nvm/nvm.sh` 후 재시도 |
| 모델 로딩 실패 | `./setup_assets.sh` 로 `workspace/models/ocr/v6` 받기 |
