#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'drone_mixformer' 2>/dev/null || true
