#!/bin/bash

./kill_robotics.sh

cd ~/demos/dx-demo-robotics

./run_demo.sh


# Run Face ID
# =================================================
#cd ~/demos/dx_demo_internal/face_recognition_demo

#./bin/face_recognition_demo \
#	-m0 assets/models/PytorchHalfpixel.dxnn \
#	-m1 assets/models/FaceAlignment.dxnn \
#	-m2 assets/models/FaceID.dxnn \
#	-c 0\
#	-re 0 \
#	-t &


# Run Pose + Segmentation
# =================================================
#cd ~/demos/dx_demo_internal/pose_seg_demo

#./bin/pose_seg_demo \
#	-m0 assets/models/YOLOV5Pose_PPU.dxnn \
#	-m1 assets/models/DDRNet_1.dxnn \
#	-p0 1 \
#	-re 0 \
#	-c 2 &

# Run Object Detection (YOLOv26-S)
# =================================================
#cd ~/dx-all-suite/dx-runtime/dx_app

#./bin/yolov26_async -m assets/models/yolo26s-1.dxnn -c 4
