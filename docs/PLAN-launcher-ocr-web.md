# PLAN: 런처 4/4/4 구성 + OCR Web 아이템 추가

- **브랜치:** `feature/launcher-ocr-web-4x4x4` (base: `DXM1-584`) — `main`에서 직접 작업하지 않음
- **작성일:** 2026-08-24

---

## 1. 목표

1. 런처 그리드를 **4열 × 3행 = 12개(4/4/4)** 로 맞춘다.
2. 12번째 아이템으로 **OCR Web 버전**(PP-OCRv5 Online Demo)을 추가한다.
3. 기존 `launcher/main.py` 및 `scripts/*.sh` 의 코드 스타일을 그대로 유지한다.

---

## 2. 참고 자료

| 구분 | 출처 | 확인한 내용 |
|------|------|-------------|
| 제품 문서 | Google Drive `All-In-One-Demo-UM-en_v0.2.pdf` | 데모 목록 및 각 데모 설명. `PaddleOCRv5 (2/2)` 페이지가 **Web(문서 입력) 버전** 화면 — NPU/CPU 선택, Image/PDF 입력, Run OCR, 결과 보기 |
| ORG | https://github.com/DEEPX-AI | `dx_app`, `dx_rt`, `dx-modelzoo`, `PP-OCRv5_Online_demo-deepx`, `PaddleOCR-deepx` 등 |
| Web UI | https://github.com/DEEPX-AI/PP-OCRv5_Online_demo-deepx (branch `deepx`) | 진입점 `app.py`, Gradio `0.0.0.0:7860`, `inbrowser=False`, `API_URL` 기본값 `http://localhost:8080/api/v1/ocr`, 의존성 `pillow==9.5.0 / requests==2.31.0 / gradio==5.30.0` |
| OCR 서버 | https://github.com/DEEPX-AI/PaddleOCR-deepx (branch `deepx`) | 진입점 `deploy/fastapi/ocr_service.py` (uvicorn), `PORT` 기본 `8080`, `HOST` 기본 `0.0.0.0`, 엔드포인트 `POST /api/v1/ocr` · `GET /health`, `SETUP_NPU=true` 로 NPU 사용, RT 튜닝값은 `.env.deepx` |
| 스크립트 스타일 | https://github.com/DEEPX-AI/dx_app `run_demo.sh` | 배너 주석 블록, `--help` usage, `realpath` 기반 경로, `scripts/color_env.sh`·`common_util.sh` 소싱 |

### 참고: 기존 코드에 이미 남아 있는 힌트

- `scripts/kill_ocr.sh` 가 이미 `ocr_service.py`, `app.py`, `chrome`, `chromium` 을 kill 대상으로 갖고 있다 → Web 버전 편성이 원래 예정되어 있었음.
- 루트 `.gitmodules` 에 `PP-OCRv5_Online_demo-deepx` / `PaddleOCR-deepx` 가 등록되어 있으나 gitlink는 추적되지 않는 **고아 상태**. (`git submodule status` 결과 없음)
- `apps/paddle-ocr/python/scripts/` 에 dx_app 유래 `color_env.sh`, `common_util.sh` 가 이미 존재.

---

## 3. 현황 분석

| 항목 | 현재 값 |
|------|---------|
| `NUM_ITEMS` | 11 |
| `GRID_COLUMNS` | 4 |
| `LAUNCHER_ITEMS` 길이 | 11 |
| 실제 배치 | 4 / 4 / 3 |

UM 문서에는 10개 기능이 기술되어 있고, 런처에는 그중 9개가 카드로 존재한다.
문서에 있으나 런처에 없는 기능:

- **Hyundai Robotics** — 앱이 리포지토리에서 제거됨 (`2c9d004 completely remove robotics app`). 에셋 `demo-robotics.png` 만 남아 있음 → 배선 불가.
- **Multi Input CLIP** — `apps/` 에 앱 없음. 에셋 `demo-clip.png` 만 남아 있음 → 배선 불가.

→ 실제로 **동작 가능한 상태로 추가할 수 있는 아이템은 OCR Web 하나**. 따라서 `11 + 1 = 12` 로 4/4/4 가 정확히 맞는다.

### 알려진 중복 (이번 작업 범위 외, 유지)

`LAUNCHER_ITEMS` 의 6번·7번 항목이 `Real-Time Road Scene Perception` 으로 완전히 동일하다.
12칸을 채우는 데 필요하고, 별도 지시가 없었으므로 **그대로 유지**한다.
(정리하려면 UM 문서 기준으로 PIDNet / YOLOPv2 / SFA3D 를 개별 카드로 분리하는 것이 자연스럽다 — 후속 과제)

---

## 4. 변경 후 배치 (4/4/4)

```
Row 1 | YOLO26-S            | Mono Depth       | OCR (Camera)      | OCR Web (신규)
Row 2 | YOLOv5S Multi (36)  | Hands Landmark   | Road Scene        | Road Scene
Row 3 | Drone Tracking      | CLIP Single      | Model Zoo         | Perf Monitoring
```

- `NUM_ITEMS`: 11 → **12**
- `GRID_COLUMNS`: **4** (유지)
- 신규 아이템은 카메라 OCR 카드 바로 뒤(index 3)에 삽입 — 동일 계열 데모를 인접 배치

---

## 5. 신규 아이템 상세

### 5.1 런처 카드

```
title        : "Optical Character Recognition (Web)"
title_i18n   : zh / ja / ko 3개 언어 (기존 항목과 동일 규칙)
image        : assets/demo-ocr-web.png
video_label  : "PP-OCRv5 Web"   → run_ocr_web.sh
camera_label : "Stop"           → kill_ocr_web.sh
```

버튼 구성은 `Model Zoo` · `Performance Monitoring` 카드와 동일한 **Start / Stop 패턴**을 따른다
(둘 다 브라우저/상주 프로세스를 띄우는 데모이므로).

### 5.2 실행 구조

```
run_ocr_web.sh
  ├─ 1) 환경 없으면 apps/paddle-ocr-web/python/build.sh 자동 호출
  │       (clone → UI venv → 서버 venv → dx_engine → .dxnn 모델 → deepx_env.sh)
  ├─ 3) OCR 서버(${DX_OCR_API_URL}) /health 확인
  │       ├─ 응답 O → 그대로 사용
  │       └─ 응답 X → 로컬 PaddleOCR-deepx 가 있으면 ocr_service.py 기동, 없으면 경고 출력 후 계속
  ├─ 4) app.py 기동 (Gradio 0.0.0.0:7860 — app.py 에 고정)
  ├─ 5) 포트 응답 대기 → open_browser.sh --no-fullscreen http://localhost:7860 (기본 브라우저, 일반 창)
  └─ 6) $DX_LAUNCHER_READY_FILE touch (런처 Wait 조기 종료)

kill_ocr_web.sh
  └─ app.py / ocr_service.py / .browser.pid 로 기록된 브라우저 창 종료
```

- OCR 서버는 별도 셋업(Docker 권장)이 필요한 **선행 조건**이다. 스크립트는 서버가 없어도 실패하지 않고, 어떤 URL을 기대하는지 명확히 안내한다.
- 서버 주소는 `config.sh` 의 신규 변수로 노출한다.

---

## 6. 파일별 작업 목록

| # | 파일 | 작업 |
|---|------|------|
| 1 | `docs/PLAN-launcher-ocr-web.md` | 본 문서 (신규) |
| 2 | `launcher/assets/demo-ocr-web.png` | UM PDF의 `PaddleOCRv5 (2/2)` 페이지를 PNG로 추출 (신규) |
| 3 | `launcher/main.py` | `NUM_ITEMS` 11→12, OCR Web 아이템 dict 삽입 |
| 4 | `scripts/run_ocr_web.sh` | 신규 |
| 5 | `scripts/kill_ocr_web.sh` | 신규 |
| 6 | `config.sh` | `DX_OCR_API_URL`, `DX_BROWSER` 추가 |
| 7 | `apps/paddle-ocr-web/README.md` | 셋업/선행조건 문서 (신규) |
| 8 | `.gitignore` | 런타임에 clone/생성되는 경로 제외 |
| 9 | `README.md` | Repository 구조 표에 `paddle-ocr-web/` 행 추가 |
| 10 | `scripts/open_browser.sh` | 기본 브라우저로 URL/로컬 HTML 열기 (신규, 공용) |
| 12 | (제거) `scripts/toggle_ocr_server.sh` | 서버 ON/OFF 전용 버튼은 삭제. Start/Stop 이 서버까지 함께 다룬다 |
| 13 | `apps/paddle-ocr-web/{build.sh,cpp/,python/}` | 다른 앱과 동일한 백엔드별 폴더 구조로 재편. `python/build.sh` 가 환경 전체를 재현 |

---

## 7. 코드 스타일 준수 규칙

**`launcher/main.py`**
- dict 키 순서: `title` → `title_i18n` → `image` → `*_label` → `*_script` → `extra_buttons` → `*_loading_sec`
- `title_i18n` 은 zh / ja / ko 순서, en 은 `title` 이 fallback
- 스크립트 경로는 `../scripts/` 상대 경로 문자열
- trailing comma 유지, 들여쓰기 4칸, 기존 주석 블록(파일 상단 설명) 수정 없음

**`scripts/*.sh`**
- `ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"` / `WORKSPACE=...` 관용구
- `config.sh` 소싱 → 동일 데모의 `kill_*.sh` 선행 호출 → 에셋 auto-download 체크 → `DX_BACKEND` 분기
- `kill_*.sh` 는 `#!/usr/bin/env bash` + `set -euo pipefail` + `pkill -TERM -f '...' 2>/dev/null || true`
- 런처가 넘기는 `--language <code>` 인자를 깨뜨리지 않는다

---

## 8. 검증

1. `python3 -m py_compile launcher/main.py`
2. `bash -n scripts/run_ocr_web.sh scripts/kill_ocr_web.sh`
3. `launcher/main.py` 를 AST 파싱해 12개 아이템 / 4-4-4 배치 · 모든 `*_script` 경로 존재 확인
4. `launcher/assets/demo-ocr-web.png` 로드 가능 여부 확인
5. `DX_BROWSER=/bin/echo scripts/open_browser.sh <url|file>` 로 브라우저 선택 경로 dry-run

실 하드웨어(DX-M1) 및 OCR 서버가 있는 환경에서의 end-to-end 동작 확인은 이 작업 범위 밖이다.

---

## 9-0. 후속 변경 (초기 계획 이후)

| 항목 | 내용 |
|------|------|
| 폴더 구조 | `apps/paddle-ocr-web/` 를 `cpp/` + `python/` 으로 분리. 업스트림에 C++ 구현이 없어 `cpp/` 는 그 사실을 기록한 README 만 둔다 |
| 환경 재현 | `python/build.sh` — clone·venv·OCR 서버·dx_engine·모델·`deepx_env.sh` 를 멱등하게 구성. `--dx_rt` / `--cpu-only` / `--clean` 지원 |

---

## 9. 리스크 / 미결 사항

| 리스크 | 대응 |
|--------|------|
| OCR 서버(PaddleOCR-deepx)는 무거운 선행 셋업(paddlepaddle, poppler, dx_engine)이 필요 | 런처 스크립트는 서버를 강제하지 않고 `/health` 체크 + 안내 메시지로 처리. 셋업 절차는 `apps/paddle-ocr-web/README.md` 에 기술 |
| Web 데모 리포지토리의 이미지가 Git LFS 포인터라 카드 이미지로 쓸 수 없음 | UM PDF 페이지를 PNG로 추출해 사용 |
| 첫 실행 시 clone + pip install 로 지연 발생 | 런처 Wait 시간은 ready-file 로 제어. README에 사전 셋업 권장 명시 |
| Road Scene 카드 중복 | 이번 범위 외 — 유지하고 본 문서에 후속 과제로 기록 |
| 타깃 장비에 chromium 이 없을 수 있음 | `scripts/open_browser.sh` 가 데스크톱 기본 브라우저를 사용. chromium/chrome `--start-fullscreen`, firefox `--kiosk`, 그 외 `xdg-open`. `DX_BROWSER` 로 강제 지정 가능. OCR Web 은 `--no-fullscreen` (일반 창) |
| OCR 서버 미설치 시 UI 에서 `ConnectionRefused` 트레이스백 | 호스트에 `local_setup.sh` 로 CPU 서버를 셋업했다. `OCR Server` 버튼으로 ON/OFF |
| NPU 모드는 `local_deepx_setup.sh` 가 DX_RT 를 빌드하므로 sudo 대화형 입력이 필요 | DX-RT 가 이미 시스템에 설치되어 있어 재빌드가 불필요했다. `pip install <dx_rt>/python_package` (`CMAKE_ARGS=-DDX_ROOT_DIR=...`) + `setup_deepx_models.sh` + `deepx_env.sh` 생성으로 **sudo 없이** NPU 구성 완료 |
| UI 에서 NPU 선택 시 `ModuleNotFoundError: dx_engine` 로 500 | 위와 동일. 해결 후 NPU 0.2s / CPU 4.4s (34 lines) 로 검증 |
| 브라우저가 이미 실행 중이면 kill 스크립트가 창을 닫지 못함 | 기동 시 기록한 `.browser.pid` 만 종료하므로 다른 창은 건드리지 않음. 그 경우 해당 탭은 수동 종료 |
| Start 가 서버까지 기동하므로 NPU 모델 preload 시간이 대기에 포함됨 | `video_loading_sec` 60 → 90 (cold start 여유). 실측 warm start 9초이며, 브라우저가 열리는 시점에 ready-file 로 Wait 를 조기 종료한다 |
