#!/bin/bash

./kill_clip.sh

cd ~/dx-demos/clip-single

source ../.venv-pyqt5/bin/activate

python3 camera-text-matcher-async-gui.py \
  --texts "A persion giving a thumbs up" \
          "A person clapping hands" \
          "A person making a hand heart" \
          "A person making a V sign with fingers" \
          "A person holding a cup" \
          "A person signaling OK with fingers" \
  --skip-frames 6

