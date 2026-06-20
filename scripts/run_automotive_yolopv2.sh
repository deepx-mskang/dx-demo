#!/bin/bash

./kill_automotive.sh

cd ~/dx-demos/automotive/yolopv2

./build/demo_dxnn_qt5_async --loop --color 2 --exit-btn
