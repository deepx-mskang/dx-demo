#!/usr/bin/env bash
set -euo pipefail

# Terminate PP-OCRv6 demo processes.

pkill -TERM -f 'cam_ppocr_v6_demo' 2>/dev/null || true
pkill -TERM -f 'ocr_service.py' 2>/dev/null || true
pkill -TERM -f 'app.py' 2>/dev/null || true
pkill -TERM -f 'chrom' 2>/dev/null || true
