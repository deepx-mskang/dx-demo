# PP-OCRv6 C++ Qt5 Demo

Qt5/OpenCV live PP-OCRv6 demo.

## Scope

- Detection + recognition only
- Text-line classification is not loaded or executed
- Detection skips boxes tilted 30 degrees or more to the left or right
- Recognition routes crops only to ratio 5, 15, or 25 models
- Models are loaded from `assets`

Default detection model: `det_v6_m_640.dxnn`

## Build

```bash
cd paddle-ocr/cam-ppocr-v6
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/cam_ppocr_v6_demo
./build/cam_ppocr_v6_demo --camera 2 --resolution 1280x720 --fps 15
./build/cam_ppocr_v6_demo --video PATH
./build/cam_ppocr_v6_demo --enable-sharpness
./build/cam_ppocr_v6_demo --sharpness strong
./build/cam_ppocr_v6_demo --exit-btn
```

You can also launch through `./run_ocr.sh`, which stops any previous demo
process before starting `build/cam_ppocr_v6_demo`.
