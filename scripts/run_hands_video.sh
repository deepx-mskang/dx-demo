#!/bin/bash

./kill_hands.sh

cd ~/dx-demos/hand-landmark

./build/hand-landmark-pose -v ~/Videos/hands.mp4 --hide-palm --exit-btn
