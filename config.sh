#!/bin/bash
# dx-demos Top-Level Configuration

# Set the camera index to use for OpenCV/Python (e.g., 0, 1)
export DX_CAMERA_IDX="0"

# Set the camera device path to use for V4L2/C++ (e.g., /dev/video0, /dev/video1)
export DX_CAMERA_DEV="/dev/video${DX_CAMERA_IDX}"
