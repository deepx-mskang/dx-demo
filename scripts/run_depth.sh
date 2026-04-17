#!/bin/bash

cd ~/dx-demos/depth

../scripts/kill_depth.sh

source .venv-depth/bin/activate

python3 demo_depth.py -m models/depth_anything_v2_vits_294x518.dxnn -s
