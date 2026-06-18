#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'yolo26s_3' 2>/dev/null || true
