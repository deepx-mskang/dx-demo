#!/usr/bin/env bash
set -euo pipefail

# Download the latest Model Zoo page as DX_ModelZoo_<YYYYMMDD>.html and repoint
# the DX_ModelZoo_latest.html symlink at it - run_modelzoo.sh opens that symlink.
#
# --no-check-certificate: the endpoint is served with DEEPX's internal CA
# (devops.dpx.ai), which is not in the system trust store.

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
URL="${DX_MODELZOO_URL:-https://modelzoo-publish-api.devops.dpx.ai/publish/html}"
DEST="${APP_DIR}/DX_ModelZoo_$(date +%Y%m%d).html"

# On failure wget leaves an empty file behind, which run_modelzoo.sh would then
# happily open as the newest page.
wget --no-check-certificate -O "${DEST}" "${URL}" || { rm -f "${DEST}"; exit 1; }

ln -sfn "$(basename "${DEST}")" "${APP_DIR}/DX_ModelZoo_latest.html"

echo "Saved $(basename "${DEST}") ($(stat -c %s "${DEST}") bytes)"
echo "DX_ModelZoo_latest.html -> $(basename "${DEST}")"
