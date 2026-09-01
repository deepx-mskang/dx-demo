#!/usr/bin/env bash
set -euo pipefail

# Serial-QR 데모 종료:
#   serial_ocr_server  (카메라 + OCR + 정적 서빙)
#   run_serial_qr.sh 가 띄운 브라우저 창 (실행 시 기록해 둔 PID)
#
# 서버는 실행 파일 경로로 매칭한다. 'serial_ocr_server' 같은 느슨한 패턴은
# 그 문자열을 커맨드라인에 포함한 셸/에디터까지 같이 죽인다.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BROWSER_PID_FILE="${ROOT_DIR}/apps/serial-qr/.browser.pid"

pkill -TERM -f 'serial-qr/cpp/build/serial_ocr_server' 2>/dev/null || true

if [ -f "${BROWSER_PID_FILE}" ]; then
    kill -TERM "$(cat "${BROWSER_PID_FILE}")" 2>/dev/null || true
    rm -f "${BROWSER_PID_FILE}"
fi
