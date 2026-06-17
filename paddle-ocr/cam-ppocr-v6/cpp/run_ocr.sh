#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/kill_ocr.sh"

"${SCRIPT_DIR}/build/cam_ppocr_v6_demo" "$@"


#./cam_ppocr_v6_demo --width 1920 --height 1080 --fps 10 --high-accuracy
