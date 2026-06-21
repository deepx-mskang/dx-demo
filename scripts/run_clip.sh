#!/bin/bash

./kill_clip.sh

cd ~/dx-demos/clip-multi

source venv-pyqt/bin/activate

python -m clip_demo_app_pyqt.dx_realtime_demo_pyqt \
  --stream 9 \
  --camera \
  --merge_central_grid 1 \
  --fullscreen_mode 1 \
  --dark_theme 1 \
  --show_each_fps_label 0 \
  --video_fps_sync_mode 0
