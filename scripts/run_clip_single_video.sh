#!/bin/bash

./kill_clip.sh

cd ~/dx-demos/clip-single

./build/camera_text_matcher_async_gui_cpp \
  --texts "A car on fire with bright flames and black smoke" \
          "People holding a gun are at the airport and a terrorist attack occurred" \
          "A person lying on the floor after falling down in a warehouse" \
          "Cars are driving on the road" \
          "Car accident occurred on the road" \
          "A massive explosion occurred in a large concrete structure" \
  --skip-frames 6 \
  --full_screen \
  --exit-btn \
  --input assets/CLIP-demo.mp4


