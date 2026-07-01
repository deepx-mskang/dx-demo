# hand-landmark

Qt5 fullscreen C++ demo for running `assets/hand-detector_192x192.dxnn`
on the DEEPX M1 NPU and drawing hand detection boxes with confidence scores.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

Webcam input:

```bash
./build/hand-landmark --camera 0
```

Video file input:

```bash
./build/hand-landmark --video /path/to/input.mp4 --loop
```

The default confidence threshold is `0.2`.

```bash
./build/hand-landmark --camera 0 --conf 0.35
```

Fullscreen is the default. Press `Esc` or `Q` to quit, and `F` to toggle
fullscreen mode.
