# CLIP Multi-Stream — Qt5 C++

This application is the multi-stream C++ version of the `clip-single` Qt5 demo.
It accepts exactly 4 or 9 GStreamer inputs, shares one asynchronous DEEPX CLIP
image encoder across all channels, and displays up to two threshold-qualified
text matches below each preview.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```bash
./build/clip_multi_cpp --config config.json --full_screen --exit-btn
```

For the ready-to-run 9-channel example:

```bash
./build/clip_multi_cpp --config config.9.json --full_screen --exit-btn
```

The wrapper script accepts a config as its first argument and forwards the
remaining CLI options:

```bash
./run_clip_multi_cpp.sh config.9.json --full_screen --exit-btn
```

## Decode-only 9-channel preview

`video_decode_preview` uses the same GStreamer pipeline construction as the
CLIP application, but does not load DXRT, ONNX Runtime, model files, the BPE
vocabulary, or any text prompts. Each tile reports its received FPS, decoded
frame count, resolution, and decoder error/stall state.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target video_decode_preview -j
./run_decode_preview.sh config.9.json --full_screen --exit-btn
```

To configure a build tree that does not require any inference dependencies:

```bash
cmake -S . -B build-preview -DCMAKE_BUILD_TYPE=Release -DBUILD_CLIP_MULTI_CPP=OFF
cmake --build build-preview -j
./build-preview/video_decode_preview --config config.9.json --full_screen --exit-btn
```

The header shows `9/9 Channels Active` when all channels delivered frames in
the latest one-second interval. Each tile reports decode FPS (`DEC`) separately
from rendered preview FPS (`DISP`). Press `Q` or `Esc` to close the preview.

Decoded frames are kept in one latest-frame slot per channel instead of being
queued into the Qt event loop. Color conversion and preview resizing run in a
shared worker pool, while the GUI thread only displays the latest processed
image at a bounded refresh rate. The default worker count is half of the
detected hardware threads, capped by the channel count.

Recommended starting point for Orange Pi 5 Plus:

```bash
./run_decode_preview.sh config.9.json --full_screen --workers 4 --display-fps 15
```

Both options are portable and can be omitted on x86_64 to use automatic worker
selection. Lowering `--display-fps` reduces GUI load without limiting decode
FPS.

The number of entries in `streams` selects the layout automatically:

- `config.json` — 4 streams, 2×2 grid
- `config.9.json` — 9 streams, 3×3 grid

## Config

Each stream contains its own source and text definitions. Every text definition
has an independent match threshold:

```json
{
  "name": "Front Camera",
  "source": "/dev/video0",
  "texts": [
    {"text": "A person is entering the building", "threshold": 0.25},
    {"text": "Smoke is visible", "threshold": 0.31}
  ]
}
```

Local video paths and `/dev/videoN` devices are converted to GStreamer
pipelines automatically. For RTSP or hardware-specific decoding, provide a
complete OpenCV-compatible pipeline instead:

For video files, `capture.fps` is applied through GStreamer `videorate` and the
appsink follows the source clock (`sync=true`), so changing the processing FPS
does not speed up or slow down playback. Live camera and URI inputs use
low-latency `sync=false` pipelines.

```json
{
  "name": "RTSP Camera",
  "pipeline": "rtspsrc location=rtsp://example/live latency=100 ! decodebin ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=2 sync=false",
  "texts": [
    {"text": "A person is present", "threshold": 0.27}
  ]
}
```
