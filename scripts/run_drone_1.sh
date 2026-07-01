#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

"$(dirname "$0")"/kill_drone.sh

cd "${ROOT_DIR}"/apps/drone


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python" ]; then
        cd python
        if [ -n "$(find . -maxdepth 1 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 1 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/drone/mixformer_sim.dxnn" --video "${WORKSPACE}/videos/common/drone-test.mp4" --exit-btn --full_screen --loop
        elif [ -n "$(find . -maxdepth 1 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 1 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/drone/mixformer_sim.dxnn" --video "${WORKSPACE}/videos/common/drone-test.mp4" --exit-btn --full_screen --loop
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
    ./build/drone_mixformer --backend dxnn --model "${WORKSPACE}/models/drone/mixformer_sim.dxnn" --video "${WORKSPACE}/videos/common/drone-test.mp4" --exit-btn --full_screen --loop

fi
