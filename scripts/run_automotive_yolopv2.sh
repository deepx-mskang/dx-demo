#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

"$(dirname "$0")"/kill_automotive.sh

cd "${ROOT_DIR}"/apps/automotive


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python/yolopv2" ]; then
        cd python/yolopv2
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/automotive/yolopv2_384x640_q-lite.dxnn" --video "${WORKSPACE}/videos/automotive/yolopv2.mp4" --exit-btn
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/automotive/yolopv2_384x640_q-lite.dxnn" --video "${WORKSPACE}/videos/automotive/yolopv2.mp4" --exit-btn
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
    cd cpp/yolopv2
    ./build/yolopv2_async --loop --color 2 --exit-btn -m "${WORKSPACE}/models/automotive/yolopv2_384x640_q-lite.dxnn" -v "${WORKSPACE}/videos/automotive/yolopv2.mp4"

fi
