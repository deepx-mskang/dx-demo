#!/bin/bash

./kill_depth.sh

source ~/demos/venv_gst_en/bin/activate

cd ~/demos/depth-anythingv2

python3 demo_depth.py -m depth_anything_v2_vits_294x518_sim_aggsv.dxnn -s
