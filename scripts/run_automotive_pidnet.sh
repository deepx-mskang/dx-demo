#!/bin/bash

./kill_automotive.sh

cd ~/dx-demos/automotive/pidnet

./build/pidnet_s_cityscapes_async -m assets/pidnet_s_cityscapes_val_fixed.dxnn -v ~/Videos/pidnet-input-video.mp4 --full_screen --config config.json --seg-palette pastel --exit-btn
