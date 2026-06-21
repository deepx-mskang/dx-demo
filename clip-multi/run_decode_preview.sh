#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.9.json"

if [[ $# -gt 0 && "${1}" != -* ]]; then
    CONFIG_PATH="${1}"
    shift
fi

exec "${SCRIPT_DIR}/build/video_decode_preview" --config "${CONFIG_PATH}" "$@"
