#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

"$(dirname "$0")"/kill_clip.sh

# Auto-download missing workspace assets
if [ ! -d "${WORKSPACE}/models" ] || [ ! -d "${WORKSPACE}/videos" ]; then
    echo "Workspace assets not found. Downloading..."
    "${ROOT_DIR}/setup_assets.sh"
fi

cd "${ROOT_DIR}"/apps/clip-single


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python" ]; then
        cd python
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" \
              --texts "A car on fire with bright flames and black smoke" \
                      "People holding a gun are at the airport and a terrorist attack occurred" \
                      "A person lying on the floor after falling down in a warehouse" \
                      "Cars are driving on the road" \
                      "Car accident occurred on the road" \
                      "A massive explosion occurred in a large concrete structure" \
              --skip-frames 6 \
              --image-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b.dxnn" \
              --text-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b-text.onnx" \
              --input "${WORKSPACE}/videos/clip/CLIP-demo.mp4"
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" \
              --texts "A car on fire with bright flames and black smoke" \
                      "People holding a gun are at the airport and a terrorist attack occurred" \
                      "A person lying on the floor after falling down in a warehouse" \
                      "Cars are driving on the road" \
                      "Car accident occurred on the road" \
                      "A massive explosion occurred in a large concrete structure" \
              --skip-frames 6 \
              --image-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b.dxnn" \
              --text-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b-text.onnx" \
              --input "${WORKSPACE}/videos/clip/CLIP-demo.mp4"
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
    ./build/camera_text_matcher_async_gui_cpp \
      --texts "A car on fire with bright flames and black smoke" \
              "People holding a gun are at the airport and a terrorist attack occurred" \
              "A person lying on the floor after falling down in a warehouse" \
              "Cars are driving on the road" \
              "Car accident occurred on the road" \
              "A massive explosion occurred in a large concrete structure" \
      --skip-frames 6 \
      --full_screen \
      --exit-btn \
      --image-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b.dxnn" \
      --text-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b-text.onnx" \
      --bpe-vocab "${ROOT_DIR}/workspace/assets/clip-single/bpe_simple_vocab_16e6.txt.gz" \
      --input "${WORKSPACE}/videos/clip/CLIP-demo.mp4"



fi
