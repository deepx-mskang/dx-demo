#!/usr/bin/env bash
set -euo pipefail

# Terminate depth demo:
#   /bin/bash /home/deepx/scripts/run_depth.sh
#   python3 demo_depth.py ...

pkill -TERM -f 'demo_depth\.py' 2>/dev/null || true
pkill -TERM -f 'demo_depth_video\.py' 2>/dev/null || true

v4l2-ctl --device /dev/video0 --set-ctrl=focus_automatic_continuous=1
#pkill -TERM -f 'run_depth\.sh' 2>/dev/null || true

# sleep 1

# pkill -KILL -f 'demo_depth\.py' 2>/dev/null || true
# pkill -KILL -f 'run_depth\.sh' 2>/dev/null || true
