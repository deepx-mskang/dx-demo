#!/bin/bash
# Serial-QR 웹 데모 실행
#
#   카메라 -> PP-OCRv6 (DX-M1 NPU) -> 시리얼 인식 -> QR 발행 -> 기기 정보 조회
#
# C++ 서버 한 프로세스가 카메라 스트리밍, OCR, 정적 파일 서빙을 모두 담당한다.

set -e

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="${ROOT_DIR}/workspace"
APP_DIR="${ROOT_DIR}/apps/serial-qr"
SERVER_BIN="${APP_DIR}/cpp/build/serial_ocr_server"
WEB_DIST="${APP_DIR}/web/dist"
BROWSER_PID_FILE="${APP_DIR}/.browser.pid"

# 최상위 설정 로드
if [ -f "${ROOT_DIR}/config.sh" ]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/config.sh"
fi

PORT="${DX_SERIAL_QR_PORT:-8090}"

"$(dirname "$0")"/kill_serial_qr.sh

# 이 데모와 기존 OCR 데모는 같은 NPU 와 같은 카메라를 쓴다. 동시에 뜨면 둘 다 실패한다.
"$(dirname "$0")"/kill_ocr.sh 2>/dev/null || true

# workspace 자산 자동 다운로드
if [ ! -d "${WORKSPACE}/models" ]; then
    echo "Workspace assets not found. Downloading..."
    "${ROOT_DIR}/setup_assets.sh"
fi

if [ ! -x "${SERVER_BIN}" ]; then
    echo "Error: ${SERVER_BIN} 가 없습니다."
    echo "       apps/serial-qr/build.sh 를 먼저 실행하세요."
    read -t 5 -p "Press enter to exit..." || true
    exit 1
fi

if [ ! -f "${WEB_DIST}/index.html" ]; then
    echo "Error: ${WEB_DIST} 가 비어 있습니다 (프론트엔드 미빌드)."
    echo "       apps/serial-qr/build.sh 를 실행하세요."
    read -t 5 -p "Press enter to exit..." || true
    exit 1
fi

# 카메라: C++/V4L2 규약대로 DX_CAMERA_DEV 를 쓰되, 없으면 인덱스로 넘어간다.
CAMERA_ARGS=(--camera "${DX_CAMERA_IDX:-0}")
if [ -n "${DX_CAMERA_DEV}" ] && [ -e "${DX_CAMERA_DEV}" ]; then
    CAMERA_ARGS=(--device "${DX_CAMERA_DEV}")
fi

echo "Starting serial-qr server on port ${PORT}..."
"${SERVER_BIN}" \
    "${CAMERA_ARGS[@]}" \
    --port "${PORT}" \
    --width 1280 --height 720 --fps 15 --crop 960 &
SERVER_PID=$!

# 서버가 뜰 때까지 대기 (NPU 모델 로딩에 시간이 걸린다)
echo -n "Waiting for OCR models to load"
READY=false
for _ in $(seq 1 90); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo
        echo "Error: 서버가 예기치 않게 종료되었습니다."
        exit 1
    fi
    if curl -fsS --max-time 2 "http://localhost:${PORT}/api/health" >/dev/null 2>&1; then
        READY=true
        echo " ready."
        break
    fi
    echo -n "."
    sleep 1
done

if [ "${READY}" != true ]; then
    echo
    echo "Error: 서버가 90초 안에 준비되지 않았습니다."
    kill "${SERVER_PID}" 2>/dev/null || true
    exit 1
fi

# 런처가 붙는 경우를 대비한 준비 완료 신호
if [ -n "${DX_LAUNCHER_READY_FILE}" ]; then
    touch "${DX_LAUNCHER_READY_FILE}" 2>/dev/null || true
fi

curl -fsS --max-time 2 "http://localhost:${PORT}/api/config" 2>/dev/null || true
echo

"${ROOT_DIR}/scripts/open_browser.sh" --no-fullscreen "http://localhost:${PORT}" &
echo $! > "${BROWSER_PID_FILE}"

wait "${SERVER_PID}"
