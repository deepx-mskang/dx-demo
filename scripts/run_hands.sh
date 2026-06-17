#!/bin/bash

./kill_hands.sh

cd ~/dx-demos/hand-landmark

./build/hand-landmark-pose -c 0 --width 1280 --height 720 --hide-palm --max-hands 10 --exit-btn
#./build/hand-landmark-pose -c 0 --width 1280 --height 720 --hide-palm --max-hands 10
