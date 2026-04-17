#!/bin/bash

cd ~/dx-demos/depth

../scripts/kill_depth.sh

source .venv-depth/bin/activate

python3 demo_depth_video.py -m models/depth_anything_v2_vits_294x518.dxnn -s --video ~/Videos/dogs.mp4
