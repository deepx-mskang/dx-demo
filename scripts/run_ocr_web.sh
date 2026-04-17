#!/bin/bash

source ~/dx-demos/paddle-ocr/.venv-ocr/bin/activate

cd ~/dx-demos/paddle-ocr/PaddleOCR-deepx/deploy/fastapi
source deepx_env.sh

./run.sh &

sleep 20

cd ~/dx-demos/paddle-ocr/PP-OCRv5_Online_demo-deepx

python3 app.py
