# depth-anythingv2

C++ async demo for running Depth Anything v2 `.dxnn` models on the DEEPX NPU
with live camera or video input and an OpenCV fullscreen viewer.

## Prerequisites

- CMake 3.14+
- C++17 compiler
- OpenCV
- DXRT / DEEPX runtime installed

If DXRT is not installed in a default search path, point CMake at it with
`DXRT_INSTALLED_DIR`.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Camera input:

```bash
./build/depth-demo \
  --model assets/depth_anything_v2_vits.dxnn
```

Camera input with side-by-side preview:

```bash
./build/depth-demo \
  --model assets/depth_anything_v2_vits.dxnn \
  --side
```

Video file input:

```bash
./build/depth-demo \
  --model assets/depth_anything_v2_vits.dxnn \
  --video /path/to/input.mp4
```

Headless run:

```bash
./build/depth-demo \
  --model assets/depth_anything_v2_vits.dxnn \
  --no-ui
```

Press `q` or `Esc` to quit when the UI is enabled.

## Options

```text
Usage: ./build/depth-demo --model <PATH> [OPTIONS]
  -m, --model <PATH>          Path to .dxnn model (required)
  -v, --video <PATH>          Video file input path (default: use camera)
  -s, --side                  Show original and depth map side by side
  -c, --camera <N>            Camera index (default: 0)
      --width <N>             Camera width (default: 640)
      --height <N>            Camera height (default: 480)
      --fps <N>               Camera FPS request (default: 30)
      --queue-size <N>        Async in-flight queue size, 1..10 (default: 10)
      --backend any|v4l2      OpenCV camera backend (default: any)
      --camera-buffer-size <N>
                              Set OpenCV camera buffer size when N > 0
      --camera-fourcc CODE    Request camera pixel format, e.g. MJPG or YUYV
      --no-ui                 Disable window rendering and UI overlays
  -h, --help                  Show this help
```

## Notes

- The demo auto-detects whether the model input is `NCHW` or `NHWC`.
- Depth output is normalized per frame and colorized with OpenCV `COLORMAP_TURBO`.
- Video input loops automatically when it reaches the end of file.
