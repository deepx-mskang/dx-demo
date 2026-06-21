#!/bin/bash

./kill_automotive.sh

cd ~/dx-demos/automotive/sfa3d

./build/sfa3d_async --full_screen --exit-btn --loop
