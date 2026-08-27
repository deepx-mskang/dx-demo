# paddle-ocr-web

PP-OCRv5 Web 데모(문서 OCR). 런처의 `Optical Character Recognition (Web)` 카드가 이 앱을 실행합니다.
카메라 실시간 OCR 데모는 `apps/paddle-ocr/` 입니다.

## 구성

| 구성 요소 | 리포지토리 | 진입점 | 포트 |
|-----------|-----------|--------|------|
| Web UI (Gradio) | [PP-OCRv5_Online_demo-deepx](https://github.com/DEEPX-AI/PP-OCRv5_Online_demo-deepx) (`deepx`) | `app.py` | 7860 |
| OCR 서버 (FastAPI) | [PaddleOCR-deepx](https://github.com/DEEPX-AI/PaddleOCR-deepx) (`deepx`) | `deploy/fastapi/ocr_service.py` | 8080 |

## 폴더 구조

다른 앱과 동일하게 백엔드별로 나뉘어 있습니다.

```
apps/paddle-ocr-web/
├── build.sh      python/build.sh 로 넘기는 진입점
├── cpp/          C++ 구현 없음 (cpp/README.md 참조)
└── python/       실제 구현 + 환경 구성
    ├── build.sh
    ├── PP-OCRv5_Online_demo-deepx/   Gradio UI    (clone, git 제외)
    ├── PaddleOCR-deepx/              FastAPI 서버 (clone, git 제외)
    └── .venv/                        UI 전용 venv (git 제외)
```

## 환경 구성

폴더를 받은 뒤 한 번만 실행하면 됩니다. 재실행해도 안전합니다.

```bash
cd apps/paddle-ocr-web
./build.sh                 # clone → venv → OCR 서버 → NPU(자동 감지)
./build.sh --cpu-only      # NPU 없이 CPU 만
./build.sh --dx_rt PATH    # dx_rt 위치 직접 지정
./build.sh --clean         # 재구성
```

`scripts/run_ocr_web.sh` 는 환경이 없으면 `python/build.sh` 를 자동 호출합니다.
자세한 내용은 `python/README.md` 를 참조하세요.

## 실행

```bash
./scripts/run_ocr_web.sh     # OCR 서버 + UI 기동, 기본 브라우저 일반 창으로 열림
./scripts/kill_ocr_web.sh    # OCR 서버 + UI + 브라우저 창 종료
```

런처 카드의 버튼 구성:

| 버튼 | 스크립트 | 동작 |
|------|----------|------|
| `PP-OCRv5 Web` | `run_ocr_web.sh` | OCR 서버 기동 → UI 기동 → 브라우저 열기 |
| `Stop` | `kill_ocr_web.sh` | OCR 서버 · UI · 브라우저 창 모두 종료 |

서버만 따로 켜고 끄려면 업스트림 스크립트를 직접 사용합니다.

```bash
cd apps/paddle-ocr-web/PaddleOCR-deepx/deploy/fastapi
./run.sh      # 서버만 기동
./stop.sh     # 서버만 종료
```

서버 주소는 루트 `config.sh` 의 `DX_OCR_API_URL` 로 지정합니다 (기본 `http://localhost:8080/api/v1/ocr`).
`run_ocr_web.sh` 는 `/health` 로 서버를 먼저 확인하고, 응답이 없으면 아래 로컬 셋업이 있을 때만 서버를 기동합니다.

브라우저는 `scripts/open_browser.sh` 가 데스크톱 기본 브라우저(`xdg-settings get default-web-browser`)로 엽니다.
OCR Web 데모는 파일 업로드·결과 확인이 필요하므로 `--no-fullscreen` 으로 **일반 창**에서 엽니다
(전체화면이 필요한 Model Zoo 는 chromium/chrome `--start-fullscreen`, firefox `--kiosk` 를 사용).
알 수 없는 브라우저는 `xdg-open` 으로 넘깁니다. 특정 브라우저를 강제하려면 `config.sh` 의 `DX_BROWSER` 를 지정하세요.

## OCR 서버 셋업 (선행 조건)

UI는 프론트엔드일 뿐이고 실제 인식은 서버에서 수행됩니다. 서버가 없으면 UI는 열리지만 요청이 실패합니다.

### 방법 1. Docker (권장)

```bash
git clone -b deepx https://github.com/DEEPX-AI/PaddleOCR-deepx.git
cd PaddleOCR-deepx/deploy/fastapi
./docker_build.sh --deepx        # NPU 모델 포함 이미지 빌드
./docker_run.sh --port 8080
curl http://localhost:8080/health
```

### 방법 2. 호스트 로컬 실행 (현재 이 리포지토리에 셋업된 방식)

`python/build.sh` 가 아래 업스트림 스크립트들을 대신 호출합니다. 직접 다루려면:

```bash
cd apps/paddle-ocr-web/python/PaddleOCR-deepx/deploy/fastapi
./local_setup.sh        # CPU: venv 생성 + paddlepaddle/paddleocr 설치 (sudo 불필요)
./run.sh                # 서버 기동 (0.0.0.0:8080)
./stop.sh               # 종료
```

`run_ocr_web.sh` 가 위 `run.sh` 를 그대로 호출하므로, `deploy/fastapi/venv` 가 있으면
런처의 `PP-OCRv5 Web` 버튼 하나로 서버까지 함께 올라옵니다.

NPU 구성 상세(왜 sudo 가 필요 없는지 포함)는 `python/README.md` 에 있습니다.

#### NPU 가속 (현재 구성됨)

`run.sh` 는 `deploy/fastapi/deepx_env.sh` 가 있으면 `SETUP_NPU=true` 로, 없으면 CPU 모드로 기동합니다.
NPU 모드에 필요한 것은 세 가지입니다.

| 필요 항목 | 확보 방법 |
|-----------|-----------|
| `dx_engine` (DX-RT Python 바인딩) | `pip install <dx_rt>/python_package` |
| `.dxnn` NPU 모델 | `./setup_deepx_models.sh --deepx-path "$(pwd)/deepx"` |
| `deepx_env.sh` (RT 튜닝값) | `local_deepx_setup.sh` 가 생성. 값은 `.env.deepx` 기준 `1 2 1 3 2 4` |

업스트림 `local_deepx_setup.sh --dx_rt <path>` 가 위 셋을 한 번에 처리하지만, 내부에서 `dx_rt/build.sh`
를 돌리며 `sudo ninja install` · `systemctl` 을 수행하므로 **sudo 비밀번호 입력이 필요**합니다.

**이미 DX-RT 가 시스템에 설치되어 있으면(`dxrt-cli -s` 동작, `/usr/local/lib/libdxrt.so` 존재)
DX_RT 재빌드는 불필요하므로 sudo 없이 구성할 수 있습니다.** 이 리포지토리는 그 방식으로 셋업되어 있습니다.

```bash
DX_RT=/home/cana/Desktop/deepx/SDK/dx-all-suite/dx-runtime/dx_rt

# 1) dx_engine 빌드·설치 (시스템 libdxrt 에 링크, sudo 불필요)
#    DX_ROOT_DIR 을 넘기지 않으면 lib/include 를 찾지 못해 실패한다
CMAKE_ARGS="-DDX_ROOT_DIR=${DX_RT}" venv/bin/pip install "${DX_RT}/python_package"

# 2) NPU 모델 다운로드 (server / mobile 세트)
./setup_deepx_models.sh --deepx-path "$(pwd)/deepx"

# 3) run.sh 로 기동 -> SETUP_NPU=true
./run.sh
```

빌드 요구사항: cmake 3.15+, g++, pybind11(pip 자동). 모델은 `deepx/engine/model_files/{server,mobile}`
에 심볼릭 링크로 배치되고, UI의 Inference Device 선택이 요청의 `deepx` 필드로 전달됩니다.

측정값 (`general_ocr_002.png`, 34 lines): **NPU 0.2s vs CPU 4.4s**

## 주요 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `DX_OCR_API_URL` | `http://localhost:8080/api/v1/ocr` | UI가 호출할 서버 엔드포인트 (`config.sh`) |
| `API_URL` | 위 값이 주입됨 | `app.py` 가 읽는 변수 |
| `SETUP_NPU` | `run.sh` 가 `deepx_env.sh` 유무로 결정 | NPU 사용 여부 |
| `DEEPX_PATH` | `deploy/fastapi/deepx` | `.dxnn` 모델·엔진 코드 위치 |
| `USE_MOBILE` | `true` (`run.sh` 기본) | mobile/server 모델 세트 선택 |
| `PORT` / `HOST` | `8080` / `0.0.0.0` | `ocr_service.py` 바인딩 |
| `DX_BROWSER` | (빈 값) | 사용할 브라우저 명령. 빈 값이면 데스크톱 기본 브라우저 |

UI 포트 7860은 `app.py` 의 `demo.launch(server_port=7860)` 에 고정되어 있어 변경할 수 없습니다.
