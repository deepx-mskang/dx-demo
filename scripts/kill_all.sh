#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

"$(dirname "$0")"/kill_clip.sh 
"$(dirname "$0")"/kill_depth.sh
"$(dirname "$0")"/kill_drone.sh
"$(dirname "$0")"/kill_hands.sh
"$(dirname "$0")"/kill_modelzoo.sh
"$(dirname "$0")"/kill_ocr.sh
"$(dirname "$0")"/kill_perf.sh
"$(dirname "$0")"/kill_yolo26.sh
"$(dirname "$0")"/kill_yolo_multi.sh
"$(dirname "$0")"/kill_automotive.sh

sleep 0.1
