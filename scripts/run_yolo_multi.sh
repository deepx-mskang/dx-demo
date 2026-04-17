#!/bin/bash

./kill_yolo_multi.sh

cd ~/dx-demos/yolo-multi
./bin/yolo_multi_demo -c config/ppu_yolo_multi_demo_36.json
