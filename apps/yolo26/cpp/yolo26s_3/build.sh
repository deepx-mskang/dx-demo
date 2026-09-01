#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"

# Walk up to the repo root instead of hardcoding a "../../../.." depth, which
# breaks silently the moment this demo moves to a different nesting level.
FETCH_MODEL=""
_dir="${SCRIPT_DIR}"
while [ "${_dir}" != "/" ]; do
    if [ -x "${_dir}/scripts/fetch_yolo26_depth_model.sh" ]; then
        FETCH_MODEL="${_dir}/scripts/fetch_yolo26_depth_model.sh"
        break
    fi
    _dir=$(dirname "${_dir}")
done

clean_build=false

while (( $# )); do
    case "$1" in
        --clean) clean_build=true; shift;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 [--clean]"; exit 1;;
    esac
done

if [ "$clean_build" = true ]; then
    rm -rf "$BUILD_DIR"
fi

# The depth panel model ships from the model zoo, not the demo_assets tarball.
# Fetching here is a convenience only - compiling needs no model, and
# run_yolo26_3{,_video}.sh fetch it again before launch - so a failure here
# (offline, model zoo down) must not abort the build.
if [ -n "${FETCH_MODEL}" ]; then
    "${FETCH_MODEL}" || echo "Warning: could not fetch the depth model; run_yolo26_3.sh will retry."
else
    echo "Warning: scripts/fetch_yolo26_depth_model.sh not found; skipping the depth model fetch."
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"
