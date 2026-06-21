#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'depth-demo' 2>/dev/null || true

