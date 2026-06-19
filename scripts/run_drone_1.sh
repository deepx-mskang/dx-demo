#!/bin/bash

./kill_drone.sh

cd ~/dx-demos/drone

./build/drone_mixformer --backend dxnn --model assets/mixformer_sim.dxnn --video ~/Videos/drone-test.mp4 --exit-btn --full_screen
