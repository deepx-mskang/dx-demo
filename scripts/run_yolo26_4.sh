#!/bin/bash

./kill_yolo26.sh

cd ~/dx-demos/yolo26

source ../.venv-pyqt5/bin/activate

python3 GUI_yolo26_all.py --width=640 --height=480 --fps 30
#python3 GUI_yolo26_all.py --width=640 --height=360
