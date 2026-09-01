#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_DIR="${ROOT_DIR}/apps/paddle-ocr-web"

# Load top-level configuration
if [ -f "${ROOT_DIR}/config.sh" ]; then
    source "${ROOT_DIR}/config.sh"
fi

"${ROOT_DIR}"/scripts/kill_ocr_web.sh

PY_DIR="${APP_DIR}/python"
WEB_DIR="${PY_DIR}/PP-OCRv5_Online_demo-deepx"
VENV_DIR="${PY_DIR}/.venv"

SERVER_DIR="${PY_DIR}/PaddleOCR-deepx"
FASTAPI_DIR="${SERVER_DIR}/deploy/fastapi"

OCR_API_URL="${DX_OCR_API_URL:-http://localhost:8080/api/v1/ocr}"
OCR_HEALTH_URL="${OCR_API_URL%/api/v1/ocr}/health"
# app.py pins server_port=7860 in demo.launch(), so the port is not configurable
WEB_URL="http://localhost:7860"

# Auto-setup on first run: build.sh clones the repos and creates both venvs
if [ ! -f "${WEB_DIR}/app.py" ] || [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "Web demo environment not found. Running python/build.sh ..."
    if ! "${PY_DIR}"/build.sh; then
        echo "Error: setup failed. See apps/paddle-ocr-web/python/README.md"
        read -t 3 -p "Press enter to exit..." || true
        exit 1
    fi
fi

# The UI is only a front-end; text recognition runs on the PaddleOCR-deepx server.
if curl -sf -m 2 "${OCR_HEALTH_URL}" > /dev/null 2>&1; then
    echo "OCR server already running at ${OCR_API_URL}"
elif [ -x "${FASTAPI_DIR}/run.sh" ] && [ -x "${FASTAPI_DIR}/venv/bin/python" ]; then
    # run.sh picks NPU up on its own when deploy/fastapi/deepx_env.sh exists
    echo "Starting OCR server ..."
    (cd "${FASTAPI_DIR}" && ./run.sh &)

    echo "Waiting for the OCR server ..."
    for _ in $(seq 1 300); do
        curl -sf -m 2 "${OCR_HEALTH_URL}" > /dev/null 2>&1 && break
        sleep 1
    done
else
    echo "Warning: no OCR server at ${OCR_API_URL} and no local PaddleOCR-deepx setup."
    echo "         The UI will start, but OCR requests will fail."
    echo "         Use the card's \"OCR Server\" button or see apps/paddle-ocr-web/README.md."
fi

echo "Running Python backend..."
cd "${WEB_DIR}"
# Absolute path so kill_ocr_web.sh can match this process unambiguously
API_URL="${OCR_API_URL}" "${VENV_DIR}"/bin/python "${WEB_DIR}"/app.py &

# app.py runs with inbrowser=False, so open the page once the port answers
echo "Waiting for ${WEB_URL} ..."
for _ in $(seq 1 120); do
    curl -sf -m 2 "${WEB_URL}" > /dev/null 2>&1 && break
    sleep 1
done

"${ROOT_DIR}"/scripts/open_browser.sh --no-fullscreen "${WEB_URL}" &
# open_browser.sh execs the browser, so $! is the browser itself
echo $! > "${APP_DIR}/.browser.pid"

# Let the launcher end its "Wait" as soon as the page is actually up
if [ -n "${DX_LAUNCHER_READY_FILE:-}" ]; then
    : > "${DX_LAUNCHER_READY_FILE}"
fi

wait
