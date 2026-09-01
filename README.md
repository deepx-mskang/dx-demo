# dx-demos

DEEPX NPU 런타임(DXRT)을 활용한 데모 모음입니다. 객체 검출, 세그멘테이션, OCR, 로보틱스, 자율주행 등 다양한 `.dxnn` 모델 예제를 포함합니다.

**지원 환경:** Ubuntu 20.04 / 22.04 / 24.04 / debian 12 / debian 13 (x86_64, aarch64)

> **⚠️ 디스플레이 해상도:** 데모 화면은 **1920 × 1080** 고정 캔버스 기준으로 그려집니다. 데모 실행 전 모니터 해상도를 1920×1080 으로 설정하세요. 다른 해상도(예: 2560×1440)에서는 풀스크린 레이아웃이 어긋나거나 화면 일부가 잘려 보일 수 있습니다.
>
> ```bash
> xrandr                                     # 출력 이름과 지원 해상도 확인
> xrandr --output HDMI-2 --mode 1920x1080    # 출력 이름은 위에서 확인한 값으로
> ```

---

## Repository 구조

데모는 모두 `apps/` 아래에 있고, 저장소 루트에는 공용 스크립트와 설정이 있습니다.

| 경로 | 설명 |
|------|------|
| `build.sh` | 최상위 빌드 스크립트 (Python 환경 + OCR Web 환경 + 모든 C++ 데모) |
| `setup_env.sh` | 공용 Python 가상환경 `.venv` 생성, `requirements.txt` 및 `dx_engine` 설치 |
| `setup_assets.sh` | `.dxnn` 모델·테스트 비디오 다운로드 (`workspace/` 생성) |
| `clean_all.sh` | C++ 빌드 산출물 정리 (`build/`, `bin/`, `cmake-build-*`) |
| `config.sh` | 카메라·브라우저 등 데모 공통 설정 |
| `launcher/` | PyQt5 기반 통합 런처 |
| `scripts/` | 런처 및 데모 실행·종료 스크립트 |
| `workspace/` | 데모 모델·비디오 (`setup_assets.sh`로 생성, git 미포함) |
| `apps/automotive/` | 자율주행 관련 데모 (PIDNet, SFA3D, YOLOPv2) |
| `apps/clip-single/` | CLIP 카메라-텍스트 매칭 (C++/Qt5) |
| `apps/depth/` | Depth Anything v2 비동기 데모 |
| `apps/drone/` | 드론 관련 데모 |
| `apps/hand-landmark/` | 손 검출 Qt5 데모 |
| `apps/paddle-ocr/` | PaddleOCR 카메라 데모 |
| `apps/paddle-ocr-web/` | PP-OCRv5 Web 문서 OCR 데모 (Gradio + FastAPI, 최상위 `build.sh` 가 환경 구성) |
| `apps/yolo26/` | YOLO26 분류·검출·세그멘테이션·포즈·뎁스 추정 |
| `apps/yolo-multi/` | 멀티 채널 YOLO 객체 검출 |
| `apps/model-zoo/` | DEEPX Model Zoo HTML 문서 |
| `apps/perf-monitor/` | CPU / NPU 성능 모니터 (PyQt5 오버레이) |

각 C++ 데모 폴더에는 `CMakeLists.txt`와 `build.sh`가 있습니다. `CMakeLists.txt`가 있는 디렉터리에서 바로 빌드할 수 있습니다.

---

## 사전 요구 사항

### 1. 시스템 패키지 (Ubuntu)

아래 명령으로 대부분의 C++ 데모에 필요한 패키지를 한 번에 설치할 수 있습니다.

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    make \
    pkg-config \
    libopencv-dev \
    qtbase5-dev \
    libssl-dev \
    zlib1g-dev \
    libx11-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libv4l-dev \
    v4l-utils \
    libcurl4-openssl-dev \
    wget
```

| 패키지 | 용도 |
|--------|------|
| `build-essential`, `cmake`, `make` | C++ 빌드 도구 (CMake 3.14 이상 필요) |
| `libopencv-dev` | OpenCV (대부분의 비전 데모) |
| `qtbase5-dev` | Qt5 Widgets GUI (카메라·뷰어 데모) |
| `libssl-dev` | OpenSSL |
| `zlib1g-dev` | ZLIB (`clip-single`) |
| `libx11-dev` | X11 풀스크린 (`depth`, 선택 사항) |
| `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev` | GStreamer (`yolo-multi`) |
| `libv4l-dev`, `v4l-utils` | USB 카메라 입력 |
| `libcurl4-openssl-dev`, `wget` | 모델·리소스 다운로드 스크립트 |

OpenCV 빌드 의존성이 부족할 경우 아래 패키지를 추가로 설치하세요.

```bash
sudo apt install -y \
    libjpeg-dev \
    libtiff5-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libxvidcore-dev \
    libavutil-dev \
    libtbb-dev \
    libeigen3-dev \
    libx264-dev \
    libgtk2.0-dev
```

### 2. DEEPX Runtime (DXRT)

대부분의 데모는 DEEPX NPU 런타임 라이브러리 **DXRT**가 필요합니다. DXRT는 apt로 제공되지 않으며, DEEPX SDK 설치 가이드에 따라 별도로 설치해야 합니다.


### 3. ONNX Runtime (`clip-single` 전용)

`clip-single`은 CLIP 텍스트 인코더를 위해 **ONNX Runtime**이 추가로 필요합니다. 시스템에 설치되어 있지 않다면 [ONNX Runtime 릴리스](https://github.com/microsoft/onnxruntime/releases)에서 Linux용 패키지를 받아 설치한 뒤, 빌드 시 루트 경로를 지정합니다.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

### 4. DEEPX NPU 하드웨어

`.dxnn` 모델을 실제로 추론하려면 DEEPX NPU가 장착된 보드와 드라이버가 필요합니다. 빌드만 검증할 때는 DXRT 헤더·라이브러리만 있으면 됩니다.

---

## 데모 에셋

`.dxnn` 모델과 테스트 비디오는 용량이 커서 git에 포함되지 않습니다. 저장소 루트에서 `setup_assets.sh`를 실행하면 [demo_assets.tar.gz](https://cs.deepx.ai/_deepx_fae_archive/demo_assets.tar.gz)를 내려받아 `workspace/`에 압축을 풉니다.

```bash
./setup_assets.sh
```

압축 해제 후 디렉터리 구조:

```text
workspace/
├── models/    # *.dxnn 모델
└── videos/    # 테스트 영상
```

| 옵션 / 환경 변수 | 설명 |
|------------------|------|
| `--force` | 기존 `workspace/`가 있어도 다시 다운로드·압축 해제 |
| `--help` | 사용법 출력 |
| `DX_DEMOS_ASSETS_URL` | 다운로드 URL 변경 (기본: `https://cs.deepx.ai/_deepx_fae_archive/demo_assets.tar.gz`) |

`workspace/`가 이미 있으면 스크립트는 건너뜁니다. 다운로드 파일은 `.cache/demo_assets.tar.gz`에 캐시됩니다. `curl` 또는 `wget`이 필요합니다.

데모 실행 전에 에셋을 먼저 받아 두세요.

---

## Model Zoo 업데이트

`apps/model-zoo/` 데모는 미리 받아 둔 정적 HTML 페이지를 브라우저로 엽니다. 최신 페이지로 갱신하려면 DEEPX 사내망에서 아래를 실행하세요. 별도 인증 토큰은 필요하지 않습니다.

```bash
./apps/model-zoo/update_modelzoo.sh
```

`apps/model-zoo/DX_ModelZoo_<YYYYMMDD>.html` 로 저장하고 `DX_ModelZoo_latest.html` 심볼릭 링크를 새 파일로 바꿉니다. 런처의 **Model Zoo → Start** 는 이 링크를 엽니다. 다운로드가 실패하면 기존 HTML 과 링크를 그대로 유지합니다.

| 환경 변수 | 설명 |
|-----------|------|
| `DX_MODELZOO_URL` | publish 엔드포인트 변경 (기본: `https://modelzoo-publish-api.devops.dpx.ai/publish/html`) |

`wget` 이 필요합니다 (`sudo apt install -y wget`). 엔드포인트는 사내 CA(`devops.dpx.ai`) 인증서를 쓰기 때문에 `--no-check-certificate` 로 받습니다.

---

## Camera Configuration (카메라 설정)

All camera-based demos (Python, C++, and JSON-configured apps) are centralized through the `config.sh` file located at the root of the repository.

If you are using a camera other than `/dev/video0` (e.g., `/dev/video1` or a USB webcam), simply update the `DX_CAMERA_IDX` in the `config.sh` file:

```bash
# dx-demos/config.sh
export DX_CAMERA_IDX="1"  # Change this to your target camera index
export DX_CAMERA_DEV="/dev/video${DX_CAMERA_IDX}"
```

- **`DX_CAMERA_IDX`**: Used primarily by Python/OpenCV backends (e.g., `0`, `1`).
- **`DX_CAMERA_DEV`**: Used by C++/V4L2/GStreamer backends (e.g., `/dev/video0`). It is automatically constructed from `DX_CAMERA_IDX`, so you typically **only need to change `DX_CAMERA_IDX`**.

By changing this one file, all launcher scripts and demos will automatically use the specified camera device without any need to recompile the C++ code or modify Python scripts.

---

## 빌드

### 최상위 빌드 (권장)

저장소 루트의 `build.sh` 하나로 Python 환경과 모든 C++ 데모를 한 번에 준비합니다.

```bash
./build.sh
```

수행 순서:

1. **Python 환경** — `setup_env.sh` 로 공용 `.venv` 를 만들고 `requirements.txt` 와 `dx_engine` 휠을 설치합니다.
2. **OCR Web 환경** — `apps/paddle-ocr-web/build.sh` 로 Gradio UI·FastAPI OCR 서버 저장소를 clone 하고 전용 venv 를 구성합니다.
3. **C++ 데모** — `apps/**/cpp/**/build.sh` 를 모두 찾아 순서대로 빌드하고, 마지막에 성공/실패 요약을 출력합니다.

| 옵션 | 설명 |
|------|------|
| `--clean` | 각 C++ 데모를 클린 빌드하고 OCR Web venv 도 새로 만듭니다 |
| `--lang cpp` | C++ 데모만 빌드 (Python·OCR Web 환경 구성 생략) |
| `--lang python` | Python·OCR Web 환경만 구성 (C++ 빌드 생략) |
| `--lang all` | 기본값. 둘 다 수행 |
| `--no-ocr-web` | OCR Web 환경 구성만 건너뜁니다 |

> **OCR Web 환경 구성은 오래 걸립니다.** 저장소 2개 clone + venv 2개 + `paddlepaddle` 설치라 네트워크 상태를 탑니다. 이미 구성돼 있으면 각 단계를 건너뛰므로 재실행은 빠릅니다. 실패하더라도 C++ 빌드는 계속 진행되고 마지막에 실패로 보고되며, `apps/paddle-ocr-web/python/build.sh` 로 따로 재시도할 수 있습니다. 이 데모가 필요 없다면 `--no-ocr-web` 을 쓰세요.

빌드 산출물을 지우려면 `./clean_all.sh` 를 실행합니다. `workspace/`, 에셋, Python 가상환경은 남깁니다.

### 개별 데모 빌드

각 C++ 데모 디렉터리에서 `build.sh`를 실행합니다.

```bash
cd apps/depth/cpp          # 예시: depth 데모
./build.sh
```

클린 빌드:

```bash
./build.sh --clean
```

각 데모 `build.sh` 동작:

1. `build/` 디렉터리 생성
2. `build/` 안에서 `cmake .. -DCMAKE_BUILD_TYPE=Release` 실행
3. `make -j$(nproc)` 로 병렬 빌드

빌드 결과 바이너리는 각 데모의 `build/` 폴더에 생성됩니다. 실행 방법은 해당 폴더의 `README.md`를 참고하세요.

### 빌드 대상 위치

| 경로 | 데모 |
|------|------|
| `apps/automotive/cpp/pidnet/` | PIDNet 세그멘테이션 |
| `apps/automotive/cpp/sfa3d/` | SFA3D 3D 검출 |
| `apps/automotive/cpp/yolopv2/` | YOLOPv2 |
| `apps/clip-single/cpp/` | CLIP 카메라-텍스트 매칭 |
| `apps/depth/cpp/` | Depth Anything v2 |
| `apps/drone/cpp/` | 드론 데모 |
| `apps/hand-landmark/cpp/` | 손 검출 |
| `apps/paddle-ocr/cpp/` | PaddleOCR 카메라 데모 |
| `apps/yolo-multi/cpp/` | 멀티 채널 YOLO |
| `apps/yolo26/cpp/yolo26s_3/` | YOLO26 4모델 (검출/포즈/세그멘테이션/깊이) |
| `apps/yolo26/cpp/yolo26s_all/` | YOLO26 전체 모델 |

`apps/paddle-ocr-web/` 는 Python 전용이라 C++ 빌드 대상이 아닙니다 (최상위 `build.sh` 의 Python 단계에서 구성).


---

## 실행 (런처)

저장소에 포함된 PyQt5 런처로 여러 데모를 GUI에서 실행할 수 있습니다.

```bash
./scripts/run_launcher.sh
```

`config.sh` 의 카메라 설정을 읽고, 실행 중인 데모를 모두 정리한 뒤 `.venv` 로 런처를 띄웁니다.
`.venv` 없이 직접 띄우려면 `sudo apt install -y python3-pyqt5` 후 `python3 launcher/main.py` 도 됩니다.

각 카드의 버튼은 상단 **Backend** 콤보에서 고른 백엔드(C++ / Python, 기본값 C++)로 실행됩니다.

자세한 설정은 [`launcher/README.md`](launcher/README.md)를 참고하세요.

---

## 문제 해결

**CMake가 DXRT를 찾지 못하는 경우**

```bash
cmake .. -DDXRT_INSTALLED_DIR=/path/to/dxrt
```

**Qt5를 찾지 못하는 경우**

```bash
sudo apt install -y qtbase5-dev
```

**카메라가 열리지 않는 경우**

```bash
sudo apt install -y v4l-utils
v4l2-ctl --list-devices    # 장치 확인
```

**`apps/automotive/cpp/sfa3d` 빌드 실패**

SFA3D는 `DXRT_ROOT` CMake 변수로 DXRT 경로를 지정해야 할 수 있습니다.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DDXRT_ROOT=/path/to/dx_rt
```

**성능 모니터링 데모에서 NPU 값이 `—` 로만 보이는 경우**

NPU 사용률·온도는 `dxtop` 과 같은 소스를 `dx_engine` 파이썬 바인딩으로 읽습니다. 바인딩이 없거나 DXRT 데몬이 응답하지 않으면 값이 비어 보입니다.

```bash
dxrt-cli -s                                        # 장치·펌웨어 응답 확인
./setup_env.sh                                     # .venv 에 dx_engine·psutil 설치
.venv/bin/python -c "import dx_engine, psutil"     # 임포트 확인
```

이 데모는 C++ 구현이 없어 런처의 **Backend 콤보 설정과 무관하게** 항상 Python GUI 모니터로 실행됩니다.

**OCR Web 데모가 시작되지 않는 경우**

환경 구성이 끝나지 않았을 수 있습니다. 아래로 다시 구성하세요 (이미 된 단계는 건너뜁니다).

```bash
./apps/paddle-ocr-web/python/build.sh
```

**데모 에셋이 없는 경우**

```bash
./setup_assets.sh
```

`workspace/models/` 또는 `workspace/videos/`가 없으면 데모 실행 시 모델·비디오를 찾지 못할 수 있습니다. 다시 받으려면 `./setup_assets.sh --force`를 사용하세요.

---

## 라이선스 및 모델

각 데모의 모델 파일(`.dxnn`)과 서드파티 라이브러리는 해당 서브 프로젝트의 라이선스를 따릅니다. 상용 배포 전 각 폴더의 README와 DEEPX SDK 이용 약관을 확인하세요.
