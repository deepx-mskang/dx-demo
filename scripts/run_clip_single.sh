#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE="$(cd "$(dirname "$0")/../workspace" && pwd)"

"$(dirname "$0")"/kill_clip.sh

cd "${ROOT_DIR}"/apps/clip-single


if [ "$DX_BACKEND" == "python" ]; then
    echo "Running Python backend..."
    if [ -d "python" ]; then
        cd python
        if [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | grep -i 'main\|demo\|gui' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" \
              --texts "A person giving a thumbs up" \
                      "A person clapping hands" \
                      "A person making a hand heart" \
                      "A person making a V sign with fingers" \
                      "A person holding a cup" \
                      "A person signaling OK with fingers" \
              --skip-frames 6 \
              --image-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b.dxnn" \
              --text-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b-text.onnx" \
              --input "/dev/video0"
        elif [ -n "$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)" ]; then
            py_file=$(find . -maxdepth 2 -name '*.py' -not -name '__init__.py' | head -n 1)
            source "${ROOT_DIR}"/.venv/bin/activate && python "$py_file" \
              --texts "A person giving a thumbs up" \
                      "A person clapping hands" \
                      "A person making a hand heart" \
                      "A person making a V sign with fingers" \
                      "A person holding a cup" \
                      "A person signaling OK with fingers" \
              --skip-frames 6 \
              --image-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b.dxnn" \
              --text-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b-text.onnx" \
              --input "/dev/video0"
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
      --texts "A persion giving a thumbs up" \
              "A person clapping hands" \
              "A person making a hand heart" \
              "A person making a V sign with fingers" \
              "A person holding a cup" \
              "A person signaling OK with fingers" \
      --skip-frames 6 \
      --full_screen \
      --image-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b.dxnn" \
      --text-encoder "${WORKSPACE}/models/clip/ViT-L-14-quickgelu-dfn2b-text.onnx" \
      --bpe-vocab "../assets/bpe_simple_vocab_16e6.txt.gz" \
      --exit-btn \
      --input "/dev/video0"



fi
