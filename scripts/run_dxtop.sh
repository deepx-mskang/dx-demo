#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

# 이 데모는 C++ 구현이 없으므로 백엔드 선택과 무관하게 Python GUI 모니터를 띄운다.
echo "Running Python GUI Performance Monitor..."

# Auto-download missing workspace assets
if [ ! -d "${WORKSPACE}/models" ] || [ ! -d "${WORKSPACE}/videos" ]; then
    echo "Workspace assets not found. Downloading..."
    "${ROOT_DIR}/setup_assets.sh"
fi

cd "${ROOT_DIR}"/apps/perf-monitor/python
source "${ROOT_DIR}"/.venv/bin/activate && python perf_monitor_design.py
