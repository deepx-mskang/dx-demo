#!/bin/bash

./kill_all.sh

cd ~/demos/dx_demo_internal/pose_seg_demo

./bin/pose_seg_demo \
	-m0 assets/models/YOLOV5Pose_PPU.dxnn \
	-m1 assets/models/DDRNet_1.dxnn \
	-p0 1 \
	-re 2 \
	-f \
	-c 0
