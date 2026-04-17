#!/usr/bin/env bash
#set -euo pipefail

# Terminate YOLO26 demo:
#   /bin/bash -c /home/deepx/scripts/run_yolo26_4.sh
#   python3 GUI_yolo26_all.py ...

pkill -TERM -f 'GUI_yolo26_all\.py' 2>/dev/null || true

v4l2-ctl --device /dev/video0 --set-ctrl=focus_automatic_continuous=1

#pkill -TERM -f 'run_yolo26_4\.sh' 2>/dev/null || true

# sleep 1

# pkill -KILL -f 'GUI_yolo26_all\.py' 2>/dev/null || true
# pkill -KILL -f 'run_yolo26_4\.sh' 2>/dev/null || true
