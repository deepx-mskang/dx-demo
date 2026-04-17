#!/bin/bash

./kill_all.sh

cd ~/demos/dx_demo_internal/face_recognition_demo

./bin/face_recognition_demo \
	-m0 assets/models/PytorchHalfpixel.dxnn \
	-m1 assets/models/FaceAlignment.dxnn \
	-m2 assets/models/FaceID.dxnn \
	-c 0 \
	-re 2 \
	-t -f
