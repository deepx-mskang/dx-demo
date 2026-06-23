#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")" && pwd)
WORKSPACE_DIR="${REPO_ROOT}/workspace"
CACHE_DIR="${REPO_ROOT}/.cache"
ARCHIVE_NAME="demo_assets.tar.gz"
ARCHIVE_PATH="${CACHE_DIR}/${ARCHIVE_NAME}"
ASSETS_URL="${DX_DEMOS_ASSETS_URL:-https://cs.deepx.ai/demo/demo_assets.tar.gz}"

FORCE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Download demo assets and extract them into workspace/.

Options:
  --force    Re-download and re-extract even if workspace/ already exists
  --help     Show this help message

Environment:
  DX_DEMOS_ASSETS_URL   Override download URL (default: ${ASSETS_URL})

Expected archive layout:
  workspace/
  ├── models/
  └── videos/
EOF
}

log() {
    echo "[setup_assets] $*"
}

die() {
    echo "[setup_assets] ERROR: $*" >&2
    exit 1
}

workspace_ready() {
    [ -d "${WORKSPACE_DIR}/models" ] && [ -d "${WORKSPACE_DIR}/videos" ]
}

download_archive() {
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 -o "${ARCHIVE_PATH}" "${ASSETS_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${ARCHIVE_PATH}" "${ASSETS_URL}"
    else
        die "curl or wget is required to download assets."
    fi
}

extract_archive() {
    local tmp_dir
    tmp_dir=$(mktemp -d)

    tar -xzf "${ARCHIVE_PATH}" -C "${tmp_dir}"

    if [ -d "${tmp_dir}/workspace" ]; then
        rm -rf "${WORKSPACE_DIR}"
        mv "${tmp_dir}/workspace" "${WORKSPACE_DIR}"
    elif [ -d "${tmp_dir}/models" ] && [ -d "${tmp_dir}/videos" ]; then
        rm -rf "${WORKSPACE_DIR}"
        mkdir -p "${WORKSPACE_DIR}"
        mv "${tmp_dir}/models" "${tmp_dir}/videos" "${WORKSPACE_DIR}/"
    else
        rm -rf "${tmp_dir}"
        die "Unexpected archive layout. Expected workspace/{models,videos}/ or {models,videos}/ at top level."
    fi

    rm -rf "${tmp_dir}"
}

while (( $# )); do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1 (try --help)"
            ;;
    esac
done

if workspace_ready && [ "${FORCE}" -eq 0 ]; then
    log "workspace/ already exists. Skipping download."
    log "Use --force to re-download and re-extract."
    exit 0
fi

mkdir -p "${CACHE_DIR}"

if [ "${FORCE}" -eq 1 ]; then
    log "Removing existing workspace/ and cached archive..."
    rm -rf "${WORKSPACE_DIR}" "${ARCHIVE_PATH}"
fi

log "Downloading ${ASSETS_URL}"
download_archive

log "Extracting ${ARCHIVE_NAME} into ${WORKSPACE_DIR}"
extract_archive

if ! workspace_ready; then
    die "Extraction finished but workspace/models/ or workspace/videos/ is missing."
fi

log "Done. Assets are available under ${WORKSPACE_DIR}"
