#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./kill_clip.sh 
./kill_depth.sh
./kill_drone.sh
./kill_hands.sh
./kill_modelzoo.sh
./kill_ocr.sh
./kill_perf.sh
./kill_yolo26.sh
./kill_yolo_multi.sh
./kill_robotics.sh
./kill_automotive.sh

sleep 0.1
