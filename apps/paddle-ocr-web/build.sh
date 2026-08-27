#!/bin/bash
set -e

# Thin entry point: this demo has a python backend only. See cpp/README.md.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec "${SCRIPT_DIR}/python/build.sh" "$@"
