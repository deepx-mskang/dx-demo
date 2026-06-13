#!/usr/bin/env bash
set -euo pipefail

# Terminate OCR demo:
#   /bin/bash /home/deepx/scripts/run_ocr.sh (may appear twice)
#   python3 demo.py
#
# Note: "python3 ... demo.py" matches any such command line; if that is too broad
# on your system, narrow the pattern in this script.

pkill -TERM -f 'cam_ocr_demo' 2>/dev/null || true
pkill -TERM -f 'ocr_service.py' 2>/dev/null || true
pkill -TERM -f 'app.py' 2>/dev/null || true
pkill -TERM -f 'chrom' 2>/dev/null || true


#sleep 1

#pkill -KILL -f 'python3.*demo-ocr\.py' 2>/dev/null || true
#pkill -KILL -f 'run_ocr\.sh' 2>/dev/null || true
