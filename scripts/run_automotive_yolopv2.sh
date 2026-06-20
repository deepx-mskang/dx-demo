#!/bin/bash

./kill_automotive.sh

cd ~/dx-demos/automotive/sfa3d

./build/demo_dxnn_async_cpp --full_screen --exit-btn --loop
