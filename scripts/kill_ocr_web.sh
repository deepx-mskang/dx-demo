#!/usr/bin/env bash
set -euo pipefail

# Terminate the PP-OCRv5 web stack:
#   app.py          (Gradio UI, port 7860)
#   ocr_service.py  (PaddleOCR-deepx FastAPI server, port 8080)
#   the browser window opened by run_ocr_web.sh (PID recorded at launch)
#
# The python processes are matched by their full script paths: a loose pattern
# such as 'app.py' also hits unrelated shells and editors. The browser is killed
# by recorded PID for the same reason - matching the URL would kill any shell
# whose command line happens to mention it, and leaves other windows alone.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BROWSER_PID_FILE="${ROOT_DIR}/apps/paddle-ocr-web/.browser.pid"

pkill -TERM -f 'PP-OCRv5_Online_demo-deepx/app\.py' 2>/dev/null || true
pkill -TERM -f 'deploy/fastapi/ocr_service\.py' 2>/dev/null || true

if [ -f "${BROWSER_PID_FILE}" ]; then
    kill -TERM "$(cat "${BROWSER_PID_FILE}")" 2>/dev/null || true
    rm -f "${BROWSER_PID_FILE}"
fi
