#!/bin/bash

./kill_clip.sh

cd ~/dx-demos/clip-multi

./build/clip_multi_cpp \
	--config config.9.json \
	--full_screen \
	--exit-btn
