#!/usr/bin/env bash
set -euo pipefail

# Terminate CLIP demos:
#   - PyQt realtime: run_clip_demo_pyqt.sh + clip_demo_app_pyqt.dx_realtime_demo_pyqt
#   - Single / terminal: x-terminal-emulator -e run_clip_single.sh, bash run_clip_single.sh,
#     camera-text-matcher-async-gui.py

pkill -TERM -f 'clip_demo_app_pyqt\.dx_realtime_demo_pyqt' 2>/dev/null || true
pkill -TERM -f 'run_clip_demo_pyqt\.sh' 2>/dev/null || true
pkill -TERM -f 'camera-text-matcher-async-gui\.py' 2>/dev/null || true

v4l2-ctl --device /dev/video0 --set-ctrl=focus_automatic_continuous=1

#pkill -TERM -f 'run_clip_single\.sh' 2>/dev/null || true

#sleep 1

#pkill -KILL -f 'clip_demo_app_pyqt\.dx_realtime_demo_pyqt' 2>/dev/null || true
#pkill -KILL -f 'run_clip_demo_pyqt\.sh' 2>/dev/null || true
#pkill -KILL -f 'camera-text-matcher-async-gui\.py' 2>/dev/null || true
#pkill -KILL -f 'run_clip_single\.sh' 2>/dev/null || true
