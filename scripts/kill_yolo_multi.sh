#!/usr/bin/env bash
set -euo pipefail

# Terminate YOLO multi demo:
#   /bin/bash /home/deepx/scripts/run_yolo_multi.sh
#   ./bin/yolo_multi_demo -c config/...

pkill -TERM -f 'yolo_multi_demo' 2>/dev/null || true
#pkill -TERM -f 'run_yolo_multi\.sh' 2>/dev/null || true

# sleep 1

# pkill -KILL -f 'yolo_multi_demo' 2>/dev/null || true
# pkill -KILL -f 'run_yolo_multi\.sh' 2>/dev/null || true
