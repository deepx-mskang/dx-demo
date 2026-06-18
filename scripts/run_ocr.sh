#!/bin/bash

./kill_ocr.sh

cd ~/dx-demos/paddle-ocr/cam-ppocr-v6

# AF Disable and set focus to 100 <- tunable value
#v4l2-ctl --device /dev/video0 --set-ctrl=focus_automatic_continuous=0
#v4l2-ctl --device /dev/video0 --set-ctrl=focus_absolute=400

./build/cam_ppocr_v6_demo --width 1280 --height 720 --fps 10 --exit-btn --enable-sharpness
