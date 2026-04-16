#!/bin/bash

./kill_yolo26.sh

cd ~/demos/yolo26

source .venv-yolo26/bin/activate

python3 GUI_yolo26_all.py --video ~/Videos/CLIP-demo.mp4
