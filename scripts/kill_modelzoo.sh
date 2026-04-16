#!/usr/bin/env bash
set -euo pipefail

# Terminate Model Zoo web stack:
#   /bin/bash /home/deepx/scripts/run_modelzoo.sh
#   /usr/lib/chromium-browser/chromium-browser ... (main + gpu/renderer/utility/zygote, same binary path)
#
# Note: Matching the Ubuntu package binary path closes every Chromium window using that install.
# If you run other sites in the same browser, they will close too.

pkill -TERM -f '/usr/lib/chromium-browser/chromium-browser' 2>/dev/null || true
pkill -TERM -f 'run_modelzoo\.sh' 2>/dev/null || true

# sleep 1

# pkill -KILL -f '/usr/lib/chromium-browser/chromium-browser' 2>/dev/null || true
# pkill -KILL -f 'run_modelzoo\.sh' 2>/dev/null || true
