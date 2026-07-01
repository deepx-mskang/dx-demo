# CLIP Camera Text Matcher — Qt5 C++

This is the C++ port of `camera-text-matcher-async-gui.py`. It uses ONNX Runtime
for the CLIP text encoder, DEEPX DXRT async inference for the image encoder, and
Qt5 Widgets for the GUI.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/camera_text_matcher_async_gui_cpp \
  --texts "A person giving a thumbs up" \
          "A person clapping hands" \
          "A person holding a cup" \
  --skip-frames 6 \
  --full_screen \
  --exit-btn
```

Video input is also supported and loops automatically:

```bash
./build/camera_text_matcher_async_gui_cpp \
  --texts "Cars are driving on the road" \
          "Car accident occurred on the road" \
  --input assets/CLIP-demo.mp4 \
  --exit-btn
```

Run with `--help` for all options.
