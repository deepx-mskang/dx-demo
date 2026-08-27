# paddle-ocr-web / python

PP-OCRv5 Web 데모의 실제 구현이 들어가는 폴더입니다.

```
python/
├── build.sh                        환경 구성 (재실행 안전)
├── PP-OCRv5_Online_demo-deepx/     Gradio Web UI  (:7860)   — build.sh 가 clone
├── PaddleOCR-deepx/                FastAPI 서버   (:8080)   — build.sh 가 clone
└── .venv/                          Web UI 전용 venv         — build.sh 가 생성
```

clone 결과물과 venv 는 git 에 포함되지 않습니다(`.gitignore`).

## 환경 구성

```bash
./build.sh                 # NPU 자동 감지 (dx_rt 경로 탐색)
./build.sh --dx_rt PATH    # dx_rt 위치를 직접 지정
./build.sh --cpu-only      # NPU 없이 CPU 만
./build.sh --clean         # venv / dx_engine 빌드 산출물 제거 후 재구성
```

`build.sh` 가 수행하는 일:

1. Web UI · OCR 서버 리포지토리 clone (branch `deepx`)
2. Web UI venv 생성 + `requirements.txt` 설치 (gradio 5.30.0 고정)
3. OCR 서버 환경 구성 — 업스트림 `deploy/fastapi/local_setup.sh` 호출 (paddlepaddle, paddleocr)
4. NPU: `dx_engine` 빌드·설치 → `.dxnn` 모델 다운로드 → `deepx_env.sh` 생성
5. 구성 결과 요약 출력

각 단계는 이미 되어 있으면 건너뛰므로 몇 번이든 다시 실행해도 됩니다.

## 실행

```bash
../../../scripts/run_ocr_web.sh     # OCR 서버 + UI 기동 후 브라우저 열기
../../../scripts/kill_ocr_web.sh    # 모두 종료
```

런처의 `Optical Character Recognition (Web)` 카드 `PP-OCRv5 Web` / `Stop` 버튼과 동일합니다.

## NPU 에 sudo 가 필요하지 않은 이유

업스트림 `local_deepx_setup.sh` 는 `dx_rt/build.sh` 로 DX-RT 를 빌드·설치하기 때문에
(`sudo ninja install`, `systemctl`) sudo 를 요구합니다. 하지만 타깃 장비에는 DX-RT 가 이미
설치되어 있으므로(`dxrt-cli -s` 동작, `/usr/local/lib/libdxrt.so` 존재) 재빌드가 필요 없고,
**빠져 있는 것은 파이썬 바인딩뿐**입니다. `build.sh` 는 그 바인딩만 venv 에 설치합니다.

```bash
CMAKE_ARGS="-DDX_ROOT_DIR=<dx_rt>" venv/bin/pip install <dx_rt>/python_package
```

`DX_ROOT_DIR` 을 넘기지 않으면 기본값이 `python_package/` 를 가리켜 `lib/include` 를 찾지 못하고
빌드가 실패합니다. DX-RT 자체가 설치되지 않은 장비에서는 `build.sh` 가 NPU 단계를 건너뛰고
CPU 모드로 구성한 뒤 안내를 출력합니다.

요구사항: cmake 3.15+, g++, python3.10+ (pybind11 은 pip 이 자동 설치)
