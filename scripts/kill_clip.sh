#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'clip_demo_app_pyqt\.dx_realtime_demo_pyqt' 2>/dev/null || true
pkill -TERM -f 'camera_text_matcher_async_gui_cpp' 2>/dev/null || true

