#!/bin/bash

./kill_ocr.sh

cd ~/demos/dx_baidu_gui

./cpp/build/cam_ocr_demo --model server
#./cpp/build/cam_ocr_demo --model mobile
