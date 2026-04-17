#!/bin/bash

cd ~/dx-demos/depth

../scripts/kill_depth.sh

source .venv-depth/bin/activate

export DXRT_DYNAMIC_CPU_THREAD=ON

python3 demo_depth.py -m models/depth_anything_v2_vits_224x224.dxnn -s
