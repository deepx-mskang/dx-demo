# dx-demos

DEEPX NPU 런타임(DXRT)을 활용한 데모 모음입니다. 객체 검출, 세그멘테이션, OCR, 로보틱스, 자율주행 등 다양한 `.dxnn` 모델 예제를 포함합니다.

**지원 환경:** Ubuntu 20.04 / 22.04 / 24.04 / debian 12 / debian 13 (x86_64, aarch64)

---

## Repository 구조

| 폴더 | 설명 |
|------|------|
| `automotive/` | 자율주행 관련 데모 (PIDNet, SFA3D, YOLOPv2) |
| `clip-single/` | CLIP 카메라-텍스트 매칭 (C++/Qt5) |
| `depth/` | Depth Anything v2 비동기 데모 |
| `drone/` | 드론 관련 데모 |
| `hand-landmark/` | 손 검출 Qt5 데모 |
| `launcher/` | PyQt5 기반 통합 런처 |
| `paddle-ocr/` | PaddleOCR 카메라 데모 |
| `yolo26/` | YOLO26 분류·검출·세그멘테이션·포즈 추정 |
| `yolo-multi/` | 멀티 채널 YOLO 객체 검출 |
| `model-zoo/` | DEEPX Model Zoo HTML 문서 |
| `perf-monitor/` | 성능 모니터링 Python 스크립트 |
| `scripts/` | 런처 및 데모 실행 스크립트 |
| `workspace/` | 데모 모델·비디오 (`setup_assets.sh`로 생성, git 미포함) |

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
| `libssl-dev` | OpenSSL (`robotics`, `encript`) |
| `zlib1g-dev` | ZLIB (`clip-single`) |
| `libx11-dev` | X11 풀스크린 (`depth`, 선택 사항) |
| `libgstreamer1.0-dev`, `libgstreamer-plugins-base1.0-dev` | GStreamer (`yolo-multi`, `robotics`) |
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

`.dxnn` 모델과 테스트 비디오는 용량이 커서 git에 포함되지 않습니다. 저장소 루트에서 `setup_assets.sh`를 실행하면 [demo_assets.tar.gz](https://cs.deepx.ai/demo/demo_assets.tar.gz)를 내려받아 `workspace/`에 압축을 풉니다.

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
| `DX_DEMOS_ASSETS_URL` | 다운로드 URL 변경 (기본: `https://cs.deepx.ai/demo/demo_assets.tar.gz`) |

`workspace/`가 이미 있으면 스크립트는 건너뜁니다. 다운로드 파일은 `.cache/demo_assets.tar.gz`에 캐시됩니다. `curl` 또는 `wget`이 필요합니다.

데모 실행 전에 에셋을 먼저 받아 두세요.

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

각 C++ 데모 디렉터리에서 `build.sh`를 실행합니다.

```bash
cd depth          # 예시: depth 데모
./build.sh
```

클린 빌드:

```bash
./build.sh --clean
```

`build.sh` 동작:

1. `build/` 디렉터리 생성
2. `build/` 안에서 `cmake .. -DCMAKE_BUILD_TYPE=Release` 실행
3. `make -j$(nproc)` 로 병렬 빌드

빌드 결과 바이너리는 각 데모의 `build/` 폴더에 생성됩니다. 실행 방법은 해당 폴더의 `README.md`를 참고하세요.

### 빌드 대상 위치

| 경로 | 데모 |
|------|------|
| `automotive/pidnet/` | PIDNet 세그멘테이션 |
| `automotive/sfa3d/` | SFA3D 3D 검출 |
| `automotive/yolopv2/` | YOLOPv2 |
| `clip-single/` | CLIP 카메라-텍스트 매칭 |
| `depth/` | Depth Anything v2 |
| `drone/` | 드론 데모 |
| `hand-landmark/` | 손 검출 |
| `paddle-ocr/cam-ppocr-v6/` | PP-OCR v6 |
| `yolo-multi/` | 멀티 채널 YOLO |
| `yolo26/yolo26s_3/` | YOLO26 3모델 |


---

## 실행 (런처)

저장소에 포함된 PyQt5 런처로 여러 데모를 GUI에서 실행할 수 있습니다.

```bash
sudo apt install -y python3-pyqt5
python3 launcher/main.py
```

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

**`automotive/sfa3d` 빌드 실패**

SFA3D는 `DXRT_ROOT` CMake 변수로 DXRT 경로를 지정해야 할 수 있습니다.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DDXRT_ROOT=/path/to/dx_rt
```

**데모 에셋이 없는 경우**

```bash
./setup_assets.sh
```

`workspace/models/` 또는 `workspace/videos/`가 없으면 데모 실행 시 모델·비디오를 찾지 못할 수 있습니다. 다시 받으려면 `./setup_assets.sh --force`를 사용하세요.

---

## 라이선스 및 모델

각 데모의 모델 파일(`.dxnn`)과 서드파티 라이브러리는 해당 서브 프로젝트의 라이선스를 따릅니다. 상용 배포 전 각 폴더의 README와 DEEPX SDK 이용 약관을 확인하세요.
