#!/bin/bash

source ../paddle-ocr/PaddleOCR-deepx/deploy/fastapi/venv/bin/activate

cd ../paddle-ocr/PaddleOCR-deepx/deploy/fastapi

source deepx_env.sh

./run.sh &

sleep 20

cd -

bash run_ocr_client.sh
