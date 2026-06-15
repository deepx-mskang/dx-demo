# Paddle OCR C++ Qt5 Demo

This is a Qt5/OpenCV C++ implementation of `../demo-ocr.py`.

## Build

```bash
sudo apt install -y qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
cd paddle-ocr/cam-ocr/cpp
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/cam_ocr_demo
./build/cam_ocr_demo --model server
./build/cam_ocr_demo --model hybrid
./build/cam_ocr_demo --camera 2 --resolution 1280x720 --fps 15
./build/cam_ocr_demo --camera 1 --width 1920 --height 1080 --fps 5
./build/cam_ocr_demo --video ../images/sample.mp4
./build/cam_ocr_demo --language korean
./build/cam_ocr_demo --enable-uvdoc
./build/cam_ocr_demo --enable-sharpness
./build/cam_ocr_demo --sharpness soft
./build/cam_ocr_demo --sharpness strong
```

Live camera input defaults to `/dev/video0`, `1280x720`, `15 FPS`, MJPG format.
Camera frames are center-cropped to `640x640` before display and OCR, so the
detection path can skip its resize step for the default live-camera workflow.
Camera options are used only for live camera input. `--video` ignores camera
device, resolution, FPS, and MJPG settings.

Sharpness is off by default. The GUI `Sharpness` checkbox turns sharpening on or
off at runtime for both preview and OCR input. CLI options let you choose the
sharpening strength:

- `--enable-sharpness` or `--sharpness` enables the default `medium` OCR-focused mode
- `--sharpness soft` applies a mild luminance-only unsharp mask
- `--sharpness medium` applies the default balance for text edges
- `--sharpness strong` applies a more aggressive luminance-only unsharp mask
- `--sharpness off` keeps sharpening disabled

The executable uses `--model mobile` by default. Model files are resolved from:

1. `../engine/model_files/mobile` or `../.temp/mobile`
2. `../engine/model_files/server` or `../.temp/server` when `--model server` is set

`--model hybrid` uses the mobile detection/classification models and the server
recognition models.

Detection logs show whether the `640` or `960` detection model is selected.

The C++ demo uses `cpp/assets/NotoSansJP-VariableFont_wght.ttf` as the default
font for the result overlay and inference result text.

Keyboard shortcuts match the Python demo for exit: `Esc` or `Q`.
