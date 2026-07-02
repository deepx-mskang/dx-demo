#!/bin/bash
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$DX_BACKEND" == "cpp" ]; then
    echo "Launching C++ dxtop in a new terminal window..."
    # 터미널 에뮬레이터(terminator)를 사용해 새 팝업 창을 띄웁니다.
    # dxtop이 종료되더라도 창이 닫히지 않도록 exec bash를 추가합니다.
    terminator -e "bash -c 'dxtop; exec bash'" &
else
    echo "Running Python GUI Performance Monitor..."
# Auto-download missing workspace assets
if [ ! -d "${WORKSPACE}/models" ] || [ ! -d "${WORKSPACE}/videos" ]; then
    echo "Workspace assets not found. Downloading..."
    "${ROOT_DIR}/setup_assets.sh"
fi

    cd "${ROOT_DIR}"/apps/perf-monitor/python
    source "${ROOT_DIR}"/.venv/bin/activate && python perf_monitor_design.py
fi
