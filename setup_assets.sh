#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="$REPO_ROOT/workspace"
ASSETS_URL="${DX_DEMOS_ASSETS_URL:-https://cs.deepx.ai/_deepx_fae_archive/demo_assets.tar.gz}"
CACHE_DIR="$REPO_ROOT/.cache"
TAR_FILE="$CACHE_DIR/demo_assets.tar.gz"

usage() {
    echo "Usage: $0 [--force] [--help]"
    echo "Downloads and extracts demo models and videos into workspace/"
    echo "  --force   Force re-download and extraction even if workspace exists"
}

FORCE=false
while (( $# )); do
    case "$1" in
        --force) FORCE=true; shift;;
        --help|-h) usage; exit 0;;
        *) echo "Unknown option: $1"; usage; exit 1;;
    esac
done

if [ "$FORCE" = false ] && [ -d "$WORKSPACE_DIR/models" ] && [ -d "$WORKSPACE_DIR/videos" ]; then
    echo "Assets already exist in workspace/. Use --force to overwrite."
    exit 0
fi

echo "========================================"
echo "Downloading Demo Assets"
echo "========================================"

mkdir -p "$CACHE_DIR"
mkdir -p "$WORKSPACE_DIR"

if [ "$FORCE" = true ] || [ ! -f "$TAR_FILE" ]; then
    echo "Downloading from $ASSETS_URL ..."
    if command -v wget &> /dev/null; then
        wget -O "$TAR_FILE" "$ASSETS_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$TAR_FILE" "$ASSETS_URL"
    else
        echo "Error: Neither wget nor curl found. Please install one of them."
        exit 1
    fi
else
    echo "Using cached archive: $TAR_FILE"
fi

echo "Extracting assets to workspace/ ..."
# The archive is packed with a top-level workspace/ directory. Strip it so the
# contents land in workspace/ instead of workspace/workspace/.
TOP_LEVEL_DIRS=$(tar -tzf "$TAR_FILE" | awk -F/ '{print $1}' | sort -u)
if [ "$TOP_LEVEL_DIRS" = "workspace" ]; then
    tar -xzf "$TAR_FILE" -C "$WORKSPACE_DIR" --strip-components=1
else
    tar -xzf "$TAR_FILE" -C "$WORKSPACE_DIR"
fi

echo "Done! Assets are ready in $WORKSPACE_DIR"
