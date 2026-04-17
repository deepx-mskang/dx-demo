#!/bin/bash

./kill_ocr.sh

cd ~/dx-demos/paddle-ocr/cam-ocr

source ../.venv-ocr/bin/activate

# AF Disable and set focus to 100 <- tunable value
v4l2-ctl --device /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl --device /dev/video0 --set-ctrl=focus_absolute=400

source set_env.sh 1 2 1 3 2 4

unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

python3 demo-ocr.py
