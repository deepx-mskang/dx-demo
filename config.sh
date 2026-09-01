#!/bin/bash
# dx-demos Top-Level Configuration

# Set the camera index to use for OpenCV/Python (e.g., 0, 1)
export DX_CAMERA_IDX="2"

# Set the camera device path to use for V4L2/C++ (e.g., /dev/video0, /dev/video1)
export DX_CAMERA_DEV="/dev/video${DX_CAMERA_IDX}"

# Set the PaddleOCR-deepx server endpoint used by the OCR Web demo
export DX_OCR_API_URL="http://localhost:8080/api/v1/ocr"

# Set the browser used by the web demos (empty = desktop default browser)
export DX_BROWSER="${DX_BROWSER:-}"

# Set the HTTP port for the Serial-QR web demo
export DX_SERIAL_QR_PORT="8090"
