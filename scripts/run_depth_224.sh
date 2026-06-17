#!/bin/bash

./kill_depth.sh

cd ~/dx-demos/depth

./build/depth-demo -m ./assets/depth_anything_v2_vits_224x224.dxnn -s --exit-btn
#./build/depth-demo -m ./assets/depth_anything_v2_vits_224x224.dxnn -s
