#!/usr/bin/env bash
set -euo pipefail

pkill -TERM -f 'pidnet_s_cityscapes_async' 2>/dev/null || true
pkill -TERM -f 'pidnet_s_cityscapes_sync' 2>/dev/null || true
pkill -TERM -f 'sfa3d_async' 2>/dev/null || true
pkill -TERM -f 'yolopv2_async' 2>/dev/null || true
