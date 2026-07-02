#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

# Load top-level configuration
if [ -f "${ROOT_DIR}/config.sh" ]; then
    source "${ROOT_DIR}/config.sh"
fi

"$(dirname "$0")"/kill_depth.sh

cd "${ROOT_DIR}"/apps/depth


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python" ]; then
        cd python
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/depth/depth_anything_v2_vits_294x518_sim.dxnn" --camera "${DX_CAMERA_IDX:-0}" --exit-btn --full_screen
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/depth/depth_anything_v2_vits_294x518_sim.dxnn" --camera "${DX_CAMERA_IDX:-0}" --exit-btn --full_screen
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
    ./build/depth-demo -m "${WORKSPACE}/models/depth/depth_anything_v2_vits_294x518_sim.dxnn" -s --exit-btn -c "${DX_CAMERA_IDX:-0}"
    #./build/depth-demo -m "${WORKSPACE}/models/depth/depth_anything_v2_vits_294x518_sim.dxnn" -s

fi
