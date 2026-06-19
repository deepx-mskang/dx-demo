#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

./kill_clip.sh 
./kill_depth.sh 
./kill_ocr.sh
./kill_yolo26.sh
./kill_yolo_multi.sh
./kill_robotics.sh

sleep 0.1
