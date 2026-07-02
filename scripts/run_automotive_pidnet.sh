#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

"$(dirname "$0")"/kill_automotive.sh

# Auto-download missing workspace assets
if [ ! -d "${WORKSPACE}/models" ] || [ ! -d "${WORKSPACE}/videos" ]; then
    echo "Workspace assets not found. Downloading..."
    "${ROOT_DIR}/setup_assets.sh"
fi

cd "${ROOT_DIR}"/apps/automotive


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python/pidnet" ]; then
        cd python/pidnet
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/automotive/pidnet_s_cityscapes_val_fixed.dxnn" --video "${WORKSPACE}/videos/common/pidnet-input-video.mp4" --exit-btn
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/automotive/pidnet_s_cityscapes_val_fixed.dxnn" --video "${WORKSPACE}/videos/common/pidnet-input-video.mp4" --exit-btn
        else
            echo "Error: Python backend not implemented for $(pwd)"
            read -t 3 -p "Press enter to exit..." || true
            exit 1
        fi
    else
        echo "Error: Python backend directory 'python' not found in $(pwd)"
        read -t 3 -p "Press enter to exit..." || true
        exit 1
    fi
else
    echo "Running C++ backend..."
    cd cpp/pidnet
    ./build/pidnet_s_cityscapes_async -m "${WORKSPACE}/models/automotive/pidnet_s_cityscapes_val_fixed.dxnn" -v "${WORKSPACE}/videos/common/pidnet-input-video.mp4" --config config.json --seg-palette pastel --exit-btn

fi
