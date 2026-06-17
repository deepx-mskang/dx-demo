#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'hand-landmark-pose' 2>/dev/null || true
