#!/bin/bash
set -e

# Fetches the YOLO26-Depth-S .dxnn used by the yolo26s_3 demo's depth panel.
#
# The model ships from the DEEPX model zoo rather than the demo_assets tarball
# that setup_assets.sh unpacks, so it is pulled separately. Both backends need
# it: the C++ build.sh calls this at build time, and run_yolo26_3*.sh calls it
# before launch so the Python backend (which never runs build.sh) gets it too.
#
# Safe to re-run; the download is skipped when any yolo26-depth-*.dxnn is
# already in place.
#
# Usage: ./fetch_yolo26_depth_model.sh [--force] [--dest DIR] [--help]

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_URL="${DX_YOLO26_DEPTH_MODEL_URL:-https://sdk.deepx.ai/modelzoo/q-lite-dxnn/2_4_0/yolo26-depth-s_768x768.dxnn}"
DEST_DIR="${ROOT_DIR}/workspace/models/depth"
force=false

usage() {
    echo "Usage: $0 [--force] [--dest DIR]"
    echo "  --force      Re-download even when a yolo26-depth-*.dxnn is present"
    echo "  --dest DIR   Target directory (default: workspace/models/depth)"
    echo
    echo "Env: DX_YOLO26_DEPTH_MODEL_URL overrides the download URL."
}

while (( $# )); do
    case "$1" in
        --force) force=true; shift;;
        --dest) DEST_DIR="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        *) echo "Unknown argument: $1"; usage; exit 1;;
    esac
done

MODEL_PATH="${DEST_DIR}/$(basename "${MODEL_URL}")"

# Any yolo26-depth-*.dxnn counts: the demo resolves the model by glob, so a
# differently named build of the same model is honoured instead of re-fetched.
existing="$(find "${DEST_DIR}" -maxdepth 1 -name 'yolo26-depth-*.dxnn' 2>/dev/null | head -n 1)"
if [ "$force" = false ] && [ -n "${existing}" ]; then
    echo "Already present: $(basename "${existing}")"
    exit 0
fi

mkdir -p "${DEST_DIR}"
echo "Downloading YOLO26-Depth-S model ..."
echo "  from ${MODEL_URL}"
echo "  to   ${MODEL_PATH}"

# Download to a temp file so an interrupted transfer never leaves a truncated
# .dxnn behind that the glob check above would then accept as valid.
TMP_PATH="${MODEL_PATH}.partial"
rm -f "${TMP_PATH}"

# set -e aborts on a failed download before any cleanup below would run, so the
# trap is what stops a half-written .partial from littering the models dir.
trap 'rm -f "${TMP_PATH}"' EXIT

if command -v wget &> /dev/null; then
    wget -O "${TMP_PATH}" "${MODEL_URL}"
elif command -v curl &> /dev/null; then
    curl -fL -o "${TMP_PATH}" "${MODEL_URL}"
else
    echo "Error: neither wget nor curl found. Please install one of them."
    exit 1
fi

if [ ! -s "${TMP_PATH}" ]; then
    rm -f "${TMP_PATH}"
    echo "Error: download produced an empty file."
    exit 1
fi

mv "${TMP_PATH}" "${MODEL_PATH}"
echo "Done: ${MODEL_PATH}"
