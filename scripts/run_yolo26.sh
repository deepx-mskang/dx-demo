#!/bin/bash

#lxterminal -e "bash -c 'source dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate && cd dx-all-suite/dx-runtime/dx_app/src/python_example/object_detection/yolov26 && python3 yolov26_multi.py --model1 yolo26s-1.dxnn --model2 yolo26s-pose.dxnn --model3 yolo26s-seg.dxnn --camera 0; exec bash'"

source ~/dx-all-suite/dx-runtime/venv-dx-runtime/bin/activate

cd ~/dx-all-suite/dx-runtime/dx_app/src/python_example/object_detection/yolov26

python3 yolov26_multi_faster.py \
	--model1 yolo26s-1.dxnn \
	--model2 yolo26s-pose.dxnn \
	--model3 yolo26s-seg.dxnn \
	--camera 0
