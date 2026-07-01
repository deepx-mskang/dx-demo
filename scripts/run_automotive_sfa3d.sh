#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

language="en"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --language)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --language" >&2
                exit 2
            fi
            language="$2"
            shift 2
            ;;
        --language=*)
            language="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

"$(dirname "$0")"/kill_automotive.sh

cd "${ROOT_DIR}"/apps/automotive


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python/sfa3d" ]; then
        cd python/sfa3d
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/automotive/sfa3d_608x608_q-lite.dxnn" --exit-btn
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" --backend dxnn --model "${WORKSPACE}/models/automotive/sfa3d_608x608_q-lite.dxnn" --exit-btn
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
    cd cpp/sfa3d
    ./build/sfa3d_async --exit-btn --loop --precompute-bev --pretrained_path "${WORKSPACE}/models/automotive/sfa3d_608x608_q-lite.dxnn"  --language "$language"

fi
