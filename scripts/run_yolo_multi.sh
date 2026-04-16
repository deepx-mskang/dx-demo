#!/bin/bash

#terminator -e "bash -c 'cd ~/dx-all-suite/dx-runtime/dx_app && ./bin/yolo_multi -c example/yolo_multi/ppu_yolo_multi_demo_36.json; exec bash'"

#cd ~/dx-all-suite/dx-runtime/dx_app
#./bin/yolo_multi -c example/yolo_multi/ppu_yolo_multi_demo_36.json


./kill_yolo_multi.sh

cd ~/demos/dx_demo_internal/yolo_multi_demo
./bin/yolo_multi_demo -c config/ppu_yolo_multi_demo_36.json
