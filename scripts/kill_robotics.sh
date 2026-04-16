#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'robotics_demo' 2>/dev/null || true

v4l2-ctl --device /dev/video0 --set-ctrl=focus_automatic_continuous=1
