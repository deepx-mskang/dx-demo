#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

# Load top-level configuration
if [ -f "${ROOT_DIR}/config.sh" ]; then
    source "${ROOT_DIR}/config.sh"
fi

"$(dirname "$0")"/kill_yolo26.sh

# Auto-download missing workspace assets
if [ ! -d "${WORKSPACE}/models" ] || [ ! -d "${WORKSPACE}/videos" ]; then
    echo "Workspace assets not found. Downloading..."
    "${ROOT_DIR}/setup_assets.sh"
fi

cd "${ROOT_DIR}"/apps/yolo26


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python" ]; then
        cd python
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" \
            	--model "${WORKSPACE}/models/common/yolo26s.dxnn" \
            	--model-pose "${WORKSPACE}/models/common/yolo26s-pose.dxnn" \
            	--model-seg "${WORKSPACE}/models/common/yolo26s-seg.dxnn" \
            	--demo-image "${ROOT_DIR}/workspace/assets/yolo26/yolo26-demo.png" \
            	-v "${DX_CAMERA_IDX:-0}"
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" \
            	--model "${WORKSPACE}/models/common/yolo26s.dxnn" \
            	--model-pose "${WORKSPACE}/models/common/yolo26s-pose.dxnn" \
            	--model-seg "${WORKSPACE}/models/common/yolo26s-seg.dxnn" \
            	--demo-image "${ROOT_DIR}/workspace/assets/yolo26/yolo26-demo.png" \
            	-v "${DX_CAMERA_IDX:-0}"
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
    cd cpp
    ./yolo26s_3/build/yolo26s_3 \
    	--model "${WORKSPACE}/models/common/yolo26s.dxnn" \
    	--model-pose "${WORKSPACE}/models/common/yolo26s-pose.dxnn" \
    	--model-seg "${WORKSPACE}/models/common/yolo26s-seg.dxnn" \
    	--demo-image "${ROOT_DIR}/workspace/assets/yolo26/yolo26-demo.png" \
    	--device "${DX_CAMERA_DEV:-/dev/video0}" \
    	--exit-btn

fi
