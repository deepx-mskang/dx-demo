#!/bin/bash

./kill_depth.sh

cd ~/dx-demos/depth

./build/depth-demo -m assets/depth_anything_v2_vits_294x518_sim.dxnn -v ~/Videos/dogs.mp4  -s --exit-btn
#./build/depth-demo -m assets/depth_anything_v2_vits_294x518_sim.dxnn -v ~/Videos/dogs.mp4  -s
