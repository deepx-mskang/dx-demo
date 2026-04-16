#!/bin/bash

./kill_clip.sh

cd /home/deepx/demos/dx-clip-starter

source .venv-clip/bin/activate

python3 camera-text-matcher-async-gui.py \
  --texts "A car on fire with bright flames and black smoke" \
          "People holding a gun are at the airport and a terrorist attack occurred" \
          "A person lying on the floor after falling down in a warehouse" \
          "Cars are driving on the road" \
          "Car accident occurred on the road" \
          "A massive explosion occurred in a large concrete structure" \
  --skip-frames 6 \
  --input assets/CLIP-demo.mp4


