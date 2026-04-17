"""
YOLOv26 multi-model demo (PyQt5 GUI): one camera input at 15 FPS, 4 independent panels.
- Panel 1: object detection (YOLOv26) + FPS overlay
- Panel 2: pose (YOLOv26Pose) + FPS overlay
- Panel 3: segmentation (YOLOv26Seg) + FPS overlay
- Panel 4: static DEEPX / Ultralytics image (no FPS)

Each panel updates on its own schedule via queued signals (no single combined imshow).

Performance notes (why FPS can sit below camera 15):
- One render thread serializes all three panels; Seg drawing is heavy and blocks OD/Pose updates.
- PyQt queued signals copy payload; QLabel + QPixmap conversion on the GUI thread is pixel-bound.
Mitigations in this file: per-panel render threads, drain queues to the latest frame only, optional
--display-scale to shrink buffers sent to Qt.

Use --show to open the PyQt window; without --show the app runs headless and prints per-panel FPS
to stdout once per second.

Inference layout: three parallel Python pipelines (each: preprocess → run_async/wait → postprocess →
optional render). They do not share Python queues between models. Each model has its own
InferenceEngine instance; DX-RT releases the GIL during native run/wait, but multiple engines
often contend on the same NPU/accelerator—logical independence does not guarantee isolated HW throughput.

Headless FPS counts completions after postprocess (no frame copy, draw_detections, or Qt). With --show,
on-screen FPS includes drawing + display path, so it is usually lower than headless for the same input.

Camera: use --gst-camera with --camera for a GStreamer MJPEG pipeline (mppjpegdec if present, else jpegdec)
→ BGR appsink; tune with --gst-device, --gst-width, --gst-height, --gst-fps. Requires OpenCV + GStreamer.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from dx_engine import Configuration
from packaging import version
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QWidget

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.performance_summary import print_async_performance_summary

from yolov26_multi_faster import (
    _draw_label,
    convert_to_original_coordinates_pose_with_params,
    convert_to_original_coordinates_seg_with_params,
    convert_to_original_coordinates_with_params,
)

try:
    from object_detection.yolov26.yolov26_async import YOLOv26
except ImportError:
    from yolov26_async import YOLOv26

try:
    from object_detection.yolov26pose.yolov26pose_async import YOLOv26Pose
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from yolov26pose_async import YOLOv26Pose

try:
    from instance_segmentation.yolov26seg.yolov26seg_async import YOLOv26Seg
except ImportError:
    from yolov26seg_async import YOLOv26Seg


@dataclass(frozen=True)
class GstCameraConfig:
    """V4L2 MJPEG → hw/sw JPEG decode → BGR for OpenCV (CAP_GSTREAMER)."""

    device: str = "/dev/video0"
    width: int = 640
    height: int = 480
    fps: int = 5


def _gstreamer_element_available(element: str) -> bool:
    try:
        r = subprocess.run(
            ["gst-inspect-1.0", element],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _pick_mjpeg_decoder_element() -> str:
    if _gstreamer_element_available("mppjpegdec"):
        return "mppjpegdec"
    if _gstreamer_element_available("jpegdec"):
        return "jpegdec"
    raise RuntimeError(
        "Neither mppjpegdec nor jpegdec is available. "
        "Install GStreamer MPP plugin or gst-plugins-bad (jpegdec)."
    )


def build_gstreamer_camera_pipeline(cfg: GstCameraConfig) -> Tuple[str, str]:
    """Returns (pipeline_string, decoder_name_used)."""
    dec = _pick_mjpeg_decoder_element()
    dev = cfg.device
    w, h, f = cfg.width, cfg.height, cfg.fps
    print(f"Building GStreamer pipeline for {dev} {w}x{h}@{f}fps with {dec}")
    pipeline = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,width={w},height={h},framerate={f}/1 ! "
        f"{dec} ! videoconvert ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    return pipeline, dec


def bgr_numpy_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """BGR uint8 contiguous -> QPixmap (copies pixel data on GUI thread)."""
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    qimg = QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


class PanelBridge(QObject):
    """Cross-thread: workers emit; Qt delivers slots on the GUI thread."""

    panel1_updated = pyqtSignal(object)
    panel2_updated = pyqtSignal(object)
    panel3_updated = pyqtSignal(object)
    panel4_static = pyqtSignal(object)


class FPSMeter:
    def __init__(self, alpha: float = 0.92) -> None:
        self._last: Optional[float] = None
        self._smoothed = 0.0
        self._alpha = alpha

    def tick(self) -> float:
        t = time.perf_counter()
        if self._last is not None:
            dt = t - self._last
            if dt > 1e-6:
                inst = 1.0 / dt
                self._smoothed = (
                    self._alpha * self._smoothed + (1.0 - self._alpha) * inst
                    if self._smoothed > 0
                    else inst
                )
        self._last = t
        return self._smoothed


class MultiPanelWindow(QMainWindow):
    def __init__(self, bridge: PanelBridge) -> None:
        super().__init__()
        self._request_stop: Optional[Callable[[], None]] = None
        self.setWindowTitle("YOLOv26 Multi (PyQt5) — 1:OD | 2:Pose | 3:Seg | 4:Image")
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)

        self._l1 = QLabel()
        self._l2 = QLabel()
        self._l3 = QLabel()
        self._l4 = QLabel()
        for lab in (self._l1, self._l2, self._l3, self._l4):
            lab.setAlignment(Qt.AlignCenter)
            lab.setScaledContents(True)
            lab.setMinimumSize(320, 180)
            lab.setStyleSheet("background-color: #282828;")

        grid.addWidget(self._l1, 0, 0)
        grid.addWidget(self._l2, 0, 1)
        grid.addWidget(self._l3, 1, 0)
        grid.addWidget(self._l4, 1, 1)

        bridge.panel1_updated.connect(self._on_p1)
        bridge.panel2_updated.connect(self._on_p2)
        bridge.panel3_updated.connect(self._on_p3)
        bridge.panel4_static.connect(self._on_p4)

    def set_stop_handler(self, fn: Callable[[], None]) -> None:
        self._request_stop = fn

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._request_stop is not None:
            self._request_stop()
        super().closeEvent(event)

    def _on_p1(self, arr: np.ndarray) -> None:
        self._l1.setPixmap(bgr_numpy_to_qpixmap(arr))

    def _on_p2(self, arr: np.ndarray) -> None:
        self._l2.setPixmap(bgr_numpy_to_qpixmap(arr))

    def _on_p3(self, arr: np.ndarray) -> None:
        self._l3.setPixmap(bgr_numpy_to_qpixmap(arr))

    def _on_p4(self, arr: np.ndarray) -> None:
        self._l4.setPixmap(bgr_numpy_to_qpixmap(arr))


def stream_inference_multi_gui(
    source: Union[int, str],
    model1: YOLOv26,
    model2: Union[YOLOv26, YOLOv26Pose],
    model3: YOLOv26Seg,
    bridge: Optional[PanelBridge] = None,
    panel4_image_path: Optional[str] = "image.jpg",
    register_stop: Optional[Callable[[Callable[[], None]], None]] = None,
    display_scale: float = 1.0,
    show_gui: bool = False,
    gst_camera: Optional[GstCameraConfig] = None,
) -> None:
    panel4_image: Optional[np.ndarray] = None
    if panel4_image_path and os.path.isfile(panel4_image_path):
        panel4_image = cv2.imread(panel4_image_path)
        if panel4_image is not None:
            print(f"[INFO] Panel 4 image loaded: {panel4_image_path}")
        else:
            print(f"[WARNING] Failed to load panel 4 image: {panel4_image_path}")
    elif panel4_image_path:
        print(f"[WARNING] Panel 4 image not found: {panel4_image_path}, using blank.")

    MAX_PENDING_INPUT_FRAMES = 2
    metrics: Dict[str, Any] = {
        "sum_read": 0.0,
        "sum_preprocess": 0.0,
        "sum_inference": 0.0,
        "sum_postprocess": 0.0,
        "sum_render": 0.0,
        "infer_completed": 0,
        "infer_first_ts": None,
        "infer_last_ts": None,
        "inflight_last_ts": None,
        "inflight_current": 0,
        "inflight_max": 0,
        "inflight_time_sum": 0.0,
    }
    metrics_lock = threading.Lock()
    # Per-model cumulative timings (postprocess completion); used for headless 1s [BENCH] deltas.
    bench_totals: Dict[str, Dict[str, Any]] = {
        "det": {"read": 0.0, "pre": 0.0, "inf": 0.0, "post": 0.0, "n": 0},
        "pose": {"read": 0.0, "pre": 0.0, "inf": 0.0, "post": 0.0, "n": 0},
        "seg": {"read": 0.0, "pre": 0.0, "inf": 0.0, "post": 0.0, "n": 0},
    }
    frames_read = [0]
    frames_enq = [0]
    frames_drop = [0]

    input_image_queue: "queue.Queue[tuple]" = queue.Queue()
    input_queue_1: "queue.Queue[tuple]" = queue.Queue()
    input_queue_2: "queue.Queue[tuple]" = queue.Queue()
    input_queue_3: "queue.Queue[tuple]" = queue.Queue()
    req_id_queue_1: "queue.Queue[tuple]" = queue.Queue()
    req_id_queue_2: "queue.Queue[tuple]" = queue.Queue()
    req_id_queue_3: "queue.Queue[tuple]" = queue.Queue()
    output_queue_1: "queue.Queue[tuple]" = queue.Queue()
    output_queue_2: "queue.Queue[tuple]" = queue.Queue()
    output_queue_3: "queue.Queue[tuple]" = queue.Queue()
    detections_queue_1: "queue.Queue[tuple]" = queue.Queue()
    detections_queue_2: "queue.Queue[tuple]" = queue.Queue()
    detections_queue_3: "queue.Queue[tuple]" = queue.Queue()

    stop_event = threading.Event()
    SENTINEL = object()

    def set_stop_event() -> None:
        stop_event.set()
        for q in [
            input_image_queue,
            input_queue_1,
            input_queue_2,
            input_queue_3,
            req_id_queue_1,
            req_id_queue_2,
            req_id_queue_3,
            output_queue_1,
            output_queue_2,
            output_queue_3,
            detections_queue_1,
            detections_queue_2,
            detections_queue_3,
        ]:
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass
        input_image_queue.put(SENTINEL)
        input_queue_1.put(SENTINEL)
        input_queue_2.put(SENTINEL)
        input_queue_3.put(SENTINEL)
        req_id_queue_1.put(SENTINEL)
        req_id_queue_2.put(SENTINEL)
        req_id_queue_3.put(SENTINEL)
        output_queue_1.put(SENTINEL)
        output_queue_2.put(SENTINEL)
        output_queue_3.put(SENTINEL)
        detections_queue_1.put(SENTINEL)
        detections_queue_2.put(SENTINEL)
        detections_queue_3.put(SENTINEL)

    if register_stop is not None:
        register_stop(set_stop_event)

    fps_lock = threading.Lock()
    fps_od: List[float] = [0.0]
    fps_pose: List[float] = [0.0]
    fps_seg: List[float] = [0.0]

    def _copy_bench() -> Tuple[Dict[str, Dict[str, Any]], int, int, int]:
        with metrics_lock:
            snap = {
                k: {
                    "read": float(v["read"]),
                    "pre": float(v["pre"]),
                    "inf": float(v["inf"]),
                    "post": float(v["post"]),
                    "n": int(v["n"]),
                }
                for k, v in bench_totals.items()
            }
            return snap, frames_read[0], frames_enq[0], frames_drop[0]

    def fps_log_worker() -> None:
        prev_bench, prev_fr, prev_fe, prev_fd = _copy_bench()
        while not stop_event.is_set():
            if stop_event.wait(1.0):
                break
            with fps_lock:
                print(
                    "[FPS] "
                    f"OD={fps_od[0]:6.2f} | Pose={fps_pose[0]:6.2f} | Seg={fps_seg[0]:6.2f}",
                    flush=True,
                )
            cur_bench, cur_fr, cur_fe, cur_fd = _copy_bench()
            d_fr = cur_fr - prev_fr
            d_fe = cur_fe - prev_fe
            d_fd = cur_fd - prev_fd
            print(
                "[BENCH] "
                f"cap_read={d_fr}/s enqueued={d_fe}/s dropped={d_fd} "
                f"dispatcher_q={input_image_queue.qsize()} "
                f"(dropped>0 => capture faster than pipeline accepts)",
                flush=True,
            )
            labels = ("OD(det)", "Pose", "Seg")
            keys = ("det", "pose", "seg")
            qchains = (
                (input_queue_1, req_id_queue_1, output_queue_1, detections_queue_1),
                (input_queue_2, req_id_queue_2, output_queue_2, detections_queue_2),
                (input_queue_3, req_id_queue_3, output_queue_3, detections_queue_3),
            )
            for lab, key, (iq, rq, oq, dq) in zip(labels, keys, qchains):
                p, c = prev_bench[key], cur_bench[key]
                dn = c["n"] - p["n"]
                if dn <= 0:
                    line = (
                        f"        {lab:8s} no completions in window | "
                        f"q in={iq.qsize()} async={rq.qsize()} post_in={oq.qsize()} render_in={dq.qsize()}"
                    )
                    print(line, flush=True)
                    continue
                avg_r = (c["read"] - p["read"]) / dn * 1000.0
                avg_p = (c["pre"] - p["pre"]) / dn * 1000.0
                avg_i = (c["inf"] - p["inf"]) / dn * 1000.0
                avg_o = (c["post"] - p["post"]) / dn * 1000.0
                inf_cap = 1000.0 / avg_i if avg_i > 1e-6 else 0.0
                print(
                    f"        {lab:8s} {dn:3d}/s  "
                    f"read={avg_r:5.2f}ms pre={avg_p:5.2f}ms inf={avg_i:6.2f}ms(~{inf_cap:5.1f}infFPS) post={avg_o:5.2f}ms  "
                    f"q in={iq.qsize()} async={rq.qsize()} post_in={oq.qsize()} render_in={dq.qsize()}",
                    flush=True,
                )
            print(
                "        (inf~X infFPS = 1000/avg_inference only; compare to ~35 if models were isolated.)",
                flush=True,
            )
            prev_bench, prev_fr, prev_fe, prev_fd = cur_bench, cur_fr, cur_fe, cur_fd

    log_thread: Optional[threading.Thread] = None
    if not show_gui:
        log_thread = threading.Thread(target=fps_log_worker, daemon=True)
        log_thread.start()

    ds = float(display_scale)
    if ds <= 0 or ds > 1.0:
        ds = 1.0

    layout_lock = threading.Lock()
    ref_h: List[Optional[int]] = [None]
    ref_w: List[Optional[int]] = [None]
    panel4_emitted = threading.Event()
    BLANK_COLOR = (40, 40, 40)

    def downscale_for_display(img: np.ndarray) -> np.ndarray:
        if ds >= 0.999:
            return img
        h, w = img.shape[:2]
        dw = max(1, int(round(w * ds)))
        dh = max(1, int(round(h * ds)))
        return cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)

    def emit_panel4_static(rh: int, rw: int) -> None:
        if bridge is None:
            return
        if panel4_image is not None:
            p4 = cv2.resize(
                panel4_image, (rw, rh), interpolation=cv2.INTER_LINEAR
            )
        else:
            p4 = np.zeros((rh, rw, 3), dtype=np.uint8)
            p4[:] = BLANK_COLOR
        bridge.panel4_static.emit(np.ascontiguousarray(downscale_for_display(p4)))

    def ensure_ref_dims(h: int, w: int) -> Tuple[int, int]:
        with layout_lock:
            if ref_h[0] is None:
                ref_h[0], ref_w[0] = h, w
            rh, rw = ref_h[0], ref_w[0]
        assert rh is not None and rw is not None
        if (
            show_gui
            and bridge is not None
            and not panel4_emitted.is_set()
        ):
            with layout_lock:
                if not panel4_emitted.is_set():
                    emit_panel4_static(rh, rw)
                    panel4_emitted.set()
        return rh, rw

    def get_latest_detection_item(q: "queue.Queue") -> Any:
        item = q.get()
        if item is SENTINEL:
            return item
        while True:
            try:
                nxt = q.get_nowait()
            except queue.Empty:
                break
            if nxt is SENTINEL:
                return nxt
            item = nxt
        return item

    def dispatcher_worker() -> None:
        while True:
            item = input_image_queue.get()
            if item is SENTINEL or stop_event.is_set():
                input_queue_1.put(SENTINEL)
                input_queue_2.put(SENTINEL)
                input_queue_3.put(SENTINEL)
                break
            frame_id, frame_bgr, meta = item
            input_queue_1.put((frame_id, frame_bgr, meta))
            input_queue_2.put((frame_id, frame_bgr, meta))
            input_queue_3.put((frame_id, frame_bgr, meta))

    def preprocess_worker(
        model: Union[YOLOv26, YOLOv26Pose, YOLOv26Seg],
        in_q: "queue.Queue[tuple]",
        out_q: "queue.Queue[tuple]",
    ) -> None:
        while True:
            item = in_q.get()
            if item is SENTINEL or stop_event.is_set():
                out_q.put(SENTINEL)
                break
            frame_id, frame_bgr, meta = item
            t0 = time.perf_counter()
            input_tensor = model.preprocess(frame_bgr)
            t1 = time.perf_counter()
            meta["t_preprocess"] = t1 - t0
            meta["t_run_async_start"] = t1
            meta["pad"] = model.pad
            meta["gain"] = model.gain
            meta["img_width"] = model.img_width
            meta["img_height"] = model.img_height
            req_id = model.ie.run_async([input_tensor])
            t2 = time.perf_counter()
            out_q.put((frame_id, frame_bgr, input_tensor, req_id, meta))
            with metrics_lock:
                if metrics["infer_first_ts"] is None:
                    metrics["infer_first_ts"] = t1
                if metrics["inflight_last_ts"] is None:
                    metrics["inflight_last_ts"] = t2
                else:
                    dt = t2 - metrics["inflight_last_ts"]
                    metrics["inflight_time_sum"] += metrics["inflight_current"] * dt
                    metrics["inflight_last_ts"] = t2
                metrics["inflight_current"] += 1
                if metrics["inflight_current"] > metrics["inflight_max"]:
                    metrics["inflight_max"] = metrics["inflight_current"]

    def wait_worker(
        model: Union[YOLOv26, YOLOv26Pose, YOLOv26Seg],
        in_q: "queue.Queue[tuple]",
        out_q: "queue.Queue[tuple]",
    ) -> None:
        while True:
            item = in_q.get()
            if item is SENTINEL or stop_event.is_set():
                out_q.put(SENTINEL)
                break
            frame_id, frame_bgr, input_tensor, req_id, meta = item
            output_tensors = model.ie.wait(req_id)
            t0 = time.perf_counter()
            meta["t_inference"] = t0 - meta["t_run_async_start"]
            out_q.put((frame_id, frame_bgr, output_tensors, meta))
            with metrics_lock:
                metrics["infer_last_ts"] = t0
                metrics["infer_completed"] += 1
                dt = t0 - metrics["inflight_last_ts"]
                metrics["inflight_time_sum"] += metrics["inflight_current"] * dt
                metrics["inflight_last_ts"] = t0
                metrics["inflight_current"] -= 1

    def postprocess_worker(
        model: Union[YOLOv26, YOLOv26Pose, YOLOv26Seg],
        in_q: "queue.Queue[tuple]",
        out_q: "queue.Queue[tuple]",
        model_key: str,
    ) -> None:
        while True:
            item = in_q.get()
            if item is SENTINEL or stop_event.is_set():
                out_q.put(SENTINEL)
                break
            frame_id, frame_bgr, output_tensors, meta = item
            t0 = time.perf_counter()
            result = model.postprocess(output_tensors)
            pad = meta["pad"]
            gain = meta["gain"]
            w, h = meta["img_width"], meta["img_height"]
            if isinstance(result, tuple) and len(result) == 2:
                detections, masks = result
                detections, masks = convert_to_original_coordinates_seg_with_params(
                    detections, masks, pad, gain, w, h
                )
                meta["t_postprocess"] = time.perf_counter() - t0
                out_q.put((frame_id, frame_bgr, detections, masks))
            else:
                detections = result
                if hasattr(model, "num_keypoints"):
                    detections = convert_to_original_coordinates_pose_with_params(
                        detections, pad, gain, w, h, model.num_keypoints
                    )
                else:
                    detections = convert_to_original_coordinates_with_params(
                        detections, pad, gain, w, h
                    )
                meta["t_postprocess"] = time.perf_counter() - t0
                out_q.put((frame_id, frame_bgr, detections))
            with metrics_lock:
                metrics["sum_read"] += meta["t_read"]
                metrics["sum_preprocess"] += meta["t_preprocess"]
                metrics["sum_inference"] += meta["t_inference"]
                metrics["sum_postprocess"] += meta["t_postprocess"]
                bt = bench_totals[model_key]
                bt["read"] += meta["t_read"]
                bt["pre"] += meta["t_preprocess"]
                bt["inf"] += meta["t_inference"]
                bt["post"] += meta["t_postprocess"]
                bt["n"] += 1

    def gui_render_worker_det() -> None:
        fps_m = FPSMeter()
        while True:
            if stop_event.is_set():
                break
            d1 = get_latest_detection_item(detections_queue_1)
            if d1 is SENTINEL:
                break
            # Headless: measure drain rate after postprocess only (no draw/Qt bottleneck).
            if not (show_gui and bridge is not None):
                fps = fps_m.tick()
                with fps_lock:
                    fps_od[0] = fps
                continue
            _fid, frame_bgr, det = d1
            h, w = frame_bgr.shape[:2]
            rh, rw = ensure_ref_dims(h, w)
            t0 = time.perf_counter()
            p1 = frame_bgr.copy()
            model1.draw_detections(p1, det)
            if p1.shape[:2] != (rh, rw):
                p1 = cv2.resize(p1, (rw, rh), interpolation=cv2.INTER_LINEAR)
            fps = fps_m.tick()
            with fps_lock:
                fps_od[0] = fps
            _draw_label(
                p1,
                f"FPS: {fps:.1f}",
                (10, 28),
                (255, 255, 200),
                (40, 40, 40),
            )
            _draw_label(p1, "YOLO26s Det", (10, 62), (200, 255, 200), (25, 55, 25))
            p1d = downscale_for_display(p1)
            bridge.panel1_updated.emit(np.ascontiguousarray(p1d))
            with metrics_lock:
                metrics["sum_render"] += time.perf_counter() - t0

    def gui_render_worker_pose() -> None:
        fps_m = FPSMeter()
        while True:
            if stop_event.is_set():
                break
            d2 = get_latest_detection_item(detections_queue_2)
            if d2 is SENTINEL:
                break
            if not (show_gui and bridge is not None):
                fps = fps_m.tick()
                with fps_lock:
                    fps_pose[0] = fps
                continue
            _fid, frame_bgr, det = d2
            h, w = frame_bgr.shape[:2]
            rh, rw = ensure_ref_dims(h, w)
            t0 = time.perf_counter()
            p2 = frame_bgr.copy()
            model2.draw_detections(p2, det)
            if p2.shape[:2] != (rh, rw):
                p2 = cv2.resize(p2, (rw, rh), interpolation=cv2.INTER_LINEAR)
            fps = fps_m.tick()
            with fps_lock:
                fps_pose[0] = fps
            _draw_label(
                p2,
                f"FPS: {fps:.1f}",
                (10, 28),
                (255, 255, 200),
                (40, 40, 40),
            )
            _draw_label(p2, "YOLO26s Pose", (10, 62), (200, 235, 255), (25, 45, 55))
            p2d = downscale_for_display(p2)
            bridge.panel2_updated.emit(np.ascontiguousarray(p2d))
            with metrics_lock:
                metrics["sum_render"] += time.perf_counter() - t0

    def gui_render_worker_seg() -> None:
        fps_m = FPSMeter()
        while True:
            if stop_event.is_set():
                break
            d3 = get_latest_detection_item(detections_queue_3)
            if d3 is SENTINEL:
                break
            if not (show_gui and bridge is not None):
                fps = fps_m.tick()
                with fps_lock:
                    fps_seg[0] = fps
                continue
            _fid, frame_bgr, det, masks = d3
            h, w = frame_bgr.shape[:2]
            rh, rw = ensure_ref_dims(h, w)
            t0 = time.perf_counter()
            p3 = frame_bgr.copy()
            model3.draw_detections(p3, det, masks)
            if p3.shape[:2] != (rh, rw):
                p3 = cv2.resize(p3, (rw, rh), interpolation=cv2.INTER_LINEAR)
            fps = fps_m.tick()
            with fps_lock:
                fps_seg[0] = fps
            _draw_label(
                p3,
                f"FPS: {fps:.1f}",
                (10, 28),
                (255, 255, 200),
                (40, 40, 40),
            )
            _draw_label(p3, "YOLO26s Seg", (10, 62), (255, 225, 200), (55, 35, 25))
            p3d = downscale_for_display(p3)
            bridge.panel3_updated.emit(np.ascontiguousarray(p3d))
            with metrics_lock:
                metrics["sum_render"] += time.perf_counter() - t0

    threads: List[threading.Thread] = [
        threading.Thread(target=dispatcher_worker, daemon=True, name="dispatcher"),
        threading.Thread(
            target=preprocess_worker,
            args=(model1, input_queue_1, req_id_queue_1),
            daemon=True,
            name="preprocess-det",
        ),
        threading.Thread(
            target=preprocess_worker,
            args=(model2, input_queue_2, req_id_queue_2),
            daemon=True,
            name="preprocess-pose",
        ),
        threading.Thread(
            target=preprocess_worker,
            args=(model3, input_queue_3, req_id_queue_3),
            daemon=True,
            name="preprocess-seg",
        ),
        threading.Thread(
            target=wait_worker,
            args=(model1, req_id_queue_1, output_queue_1),
            daemon=True,
            name="wait-det",
        ),
        threading.Thread(
            target=wait_worker,
            args=(model2, req_id_queue_2, output_queue_2),
            daemon=True,
            name="wait-pose",
        ),
        threading.Thread(
            target=wait_worker,
            args=(model3, req_id_queue_3, output_queue_3),
            daemon=True,
            name="wait-seg",
        ),
        threading.Thread(
            target=postprocess_worker,
            args=(model1, output_queue_1, detections_queue_1, "det"),
            daemon=True,
            name="postprocess-det",
        ),
        threading.Thread(
            target=postprocess_worker,
            args=(model2, output_queue_2, detections_queue_2, "pose"),
            daemon=True,
            name="postprocess-pose",
        ),
        threading.Thread(
            target=postprocess_worker,
            args=(model3, output_queue_3, detections_queue_3, "seg"),
            daemon=True,
            name="postprocess-seg",
        ),
        threading.Thread(
            target=gui_render_worker_det, daemon=True, name="render-det"
        ),
        threading.Thread(
            target=gui_render_worker_pose, daemon=True, name="render-pose"
        ),
        threading.Thread(
            target=gui_render_worker_seg, daemon=True, name="render-seg"
        ),
    ]

    for t in threads:
        t.start()

    width: int
    height: int
    fps_cam: float

    if gst_camera is not None:
        try:
            pipeline, dec_used = build_gstreamer_camera_pipeline(gst_camera)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            set_stop_event()
            for t in threads:
                t.join(timeout=5.0)
            raise SystemExit(1) from e
        print(
            f"[INFO] GStreamer camera: decoder={dec_used} "
            f"device={gst_camera.device} {gst_camera.width}x{gst_camera.height}@{gst_camera.fps}fps (MJPEG)"
        )
        print(f"[INFO] Pipeline: {pipeline}")
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(
                "[ERROR] GStreamer VideoCapture failed. Is OpenCV built with GStreamer "
                "(CAP_GSTREAMER)? Try: python -c \"import cv2; print(cv2.getBuildInformation())\""
            )
            set_stop_event()
            for t in threads:
                t.join(timeout=5.0)
            raise SystemExit(1)
        width = int(gst_camera.width)
        height = int(gst_camera.height)
        fps_cam = float(gst_camera.fps)
    else:
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open input source: {source}")
            set_stop_event()
            for t in threads:
                t.join(timeout=5.0)
            raise SystemExit(1)

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
        else:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_cam = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n[INFO] Input WxH: {width}x{height}, nominal FPS: {fps_cam:.2f}")
    if show_gui and ds < 1.0:
        print(
            f"[INFO] GUI display scale: {ds:.2f} (smaller Qt payload; use 1.0 for full-res labels)"
        )
    if not show_gui:
        print(
            "[INFO] Headless: [FPS] = drain rate/model; [BENCH] = per-model avg ms (last 1s) + queue depths."
        )
    print(
        "[INFO] Pipelines: 3 independent thread chains (Det / Pose / Seg), 3 InferenceEngine "
        "instances; accelerator may still serialize or share bandwidth across concurrent jobs."
    )

    cnt = 0
    start_time = time.perf_counter()
    try:
        while not stop_event.is_set():
            t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()
            meta = {"t_read": t1 - t0}
            frames_read[0] += 1

            if input_image_queue.qsize() < MAX_PENDING_INPUT_FRAMES:
                cnt += 1
                frames_enq[0] += 1
                input_image_queue.put((cnt - 1, frame_bgr, meta))
            else:
                frames_drop[0] += 1
    except Exception as e:
        print(f"\n[ERROR] Capture error: {e}")
        set_stop_event()
    finally:
        if not stop_event.is_set():
            set_stop_event()
        for t in threads:
            t.join(timeout=30.0)
        if log_thread is not None:
            log_thread.join(timeout=2.0)
        if metrics["infer_completed"] == 0:
            print("[WARNING] No frames were processed.")
        else:
            elapsed = time.perf_counter() - start_time
            print_async_performance_summary(metrics, cnt, elapsed, show_gui)
        cap.release()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv26 multi-model demo (PyQt5): independent panel updates + per-panel FPS."
    )
    parser.add_argument("--model1", type=str, required=True, help="YOLOv26 detection .dxnn")
    parser.add_argument("--model2", type=str, required=True, help="YOLOv26Pose .dxnn")
    parser.add_argument("--model3", type=str, required=True, help="YOLOv26Seg .dxnn")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Input video path.")
    group.add_argument("--camera", type=int, help="Camera index (e.g. 0).")
    group.add_argument("--rtsp", type=str, help="RTSP URL.")
    parser.add_argument(
        "--image",
        type=str,
        default="image.jpg",
        help="Panel 4 static image (default: image.jpg).",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.65,
        metavar="S",
        help=(
            "With --show: scale factor (0< S <=1) for BGR sent to Qt after drawing. "
            "Default 0.65; use 1.0 for native resolution."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open PyQt5 window. Without this flag, run headless and print FPS every 1s to stdout.",
    )
    parser.add_argument(
        "--gst-camera",
        action="store_true",
        help=(
            "Use GStreamer for --camera: v4l2src MJPEG → mppjpegdec (if available) else jpegdec → BGR appsink. "
            "Requires OpenCV built with GStreamer."
        ),
    )
    parser.add_argument(
        "--gst-device",
        type=str,
        default="/dev/video0",
        help="V4L2 device path when using --gst-camera (default: /dev/video0).",
    )
    parser.add_argument("--gst-width", type=int, default=1280, help="MJPEG width (default: 1280).")
    parser.add_argument("--gst-height", type=int, default=720, help="MJPEG height (default: 720).")
    parser.add_argument(
        "--gst-fps",
        type=int,
        default=30,
        help="Framerate numerator; pipeline uses N/1 (default: 24).",
    )
    return parser.parse_args()


def main() -> None:
    config = Configuration()
    if version.parse(config.get_version()) < version.parse("3.0.0"):
        print("[ERROR] DX-RT v3.0.0 or higher is required.")
        sys.exit(1)

    args = parse_arguments()
    for path, label in (
        (args.model1, "model1"),
        (args.model2, "model2"),
        (args.model3, "model3"),
    ):
        if not os.path.exists(path):
            print(f"[ERROR] --{label} file not found: {path}")
            sys.exit(1)
    if args.video and not os.path.exists(args.video):
        print("[ERROR] Video file does not exist.")
        sys.exit(1)
    if args.gst_camera:
        if args.camera is None:
            print("[ERROR] --gst-camera requires --camera (live camera mode).")
            sys.exit(1)
        if args.video is not None or args.rtsp is not None:
            print("[ERROR] --gst-camera cannot be used with --video or --rtsp.")
            sys.exit(1)
        if not os.path.exists(args.gst_device):
            print(f"[ERROR] V4L2 device not found: {args.gst_device}")
            sys.exit(1)

    model1 = YOLOv26(args.model1)
    model2 = YOLOv26Pose(args.model2)
    model3 = YOLOv26Seg(args.model3)

    source: Union[int, str]
    if args.video:
        source = args.video
    elif args.camera is not None:
        source = args.camera
    else:
        source = args.rtsp

    panel4_path = args.image.strip() or None

    gst_camera: Optional[GstCameraConfig] = None
    if args.gst_camera:
        gst_camera = GstCameraConfig(
            device=args.gst_device,
            width=args.gst_width,
            height=args.gst_height,
            fps=args.gst_fps,
        )

    if not args.show:
        stream_inference_multi_gui(
            source,
            model1,
            model2,
            model3,
            bridge=None,
            panel4_image_path=panel4_path,
            register_stop=None,
            display_scale=args.display_scale,
            show_gui=False,
            gst_camera=gst_camera,
        )
        return

    app = QApplication(sys.argv)
    bridge = PanelBridge()
    win = MultiPanelWindow(bridge)
    win.resize(1280, 720)
    win.show()

    def run_pipeline() -> None:
        try:
            stream_inference_multi_gui(
                source,
                model1,
                model2,
                model3,
                bridge,
                panel4_image_path=panel4_path,
                register_stop=lambda fn: win.set_stop_handler(fn),
                display_scale=args.display_scale,
                show_gui=True,
                gst_camera=gst_camera,
            )
        finally:
            app.quit()

    threading.Thread(target=run_pipeline, daemon=True).start()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    main()
