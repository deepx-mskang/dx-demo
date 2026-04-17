#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'perf_monitor_design\.py' 2>/dev/null || true
