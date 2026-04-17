#!/bin/bash

./kill_robotics.sh

cd ~/dx-demos/robotics

MODELS="assets/models"

BIN="./bin/robotics_demo"


"$BIN" \
    -m0 "${MODELS}/PytorchHalfpixel_1.dxnn" \
    -m1 "${MODELS}/FaceAlignment.dxnn" \
    -m2 "${MODELS}/FaceID_1.dxnn" \
    -mps0 "${MODELS}/YOLOV5Pose_PPU.dxnn" \
    -mps1 "${MODELS}/DDRNet_1.dxnn" \
    -md "${MODELS}/MobED_detector_80_fixed.dxnn" \
    -c

popd > /dev/null

