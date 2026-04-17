"""
YOLOv26 multi-model demo (faster display): one input stream.
- Model 1: object detection (YOLOv26)
- Model 2: pose estimation (YOLOv26Pose)
- Model 3: instance segmentation (YOLOv26Seg)
Output: 4-panel display (Detection | Pose; Segmentation | Empty).

Faster display: each panel updates as soon as its model result is ready,
without waiting for the other two. Improves perceived responsiveness.
"""
import argparse
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from dx_engine import Configuration
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.performance_summary import print_async_performance_summary

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


def _draw_label(
    img: np.ndarray,
    text: str,
    pt: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (45, 45, 45),
) -> None:
    """Lightweight: one getTextSize, one rect, one putText. No copy, no alpha."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.95, 2
    (w, h), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = pt[0], pt[1]
    pad = 5
    x1, y1 = x, y - h - pad
    x2, y2 = x + w + pad * 2, y + bl + pad
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
    cv2.putText(img, text, (x + pad, y), font, scale, text_color, thick, cv2.LINE_AA)


def convert_to_original_coordinates_with_params(
    detections: np.ndarray,
    pad: Tuple[int, int],
    gain: float,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Convert detection bbox to original image coordinates (object detection)."""
    if len(detections) == 0:
        return detections
    out = detections.copy()
    out[:, 0] = np.clip(
        (out[:, 0] - pad[1]) / gain, 0, img_width - 1
    )
    out[:, 1] = np.clip(
        (out[:, 1] - pad[0]) / gain, 0, img_height - 1
    )
    out[:, 2] = np.clip(
        (out[:, 2] - pad[1]) / gain, 0, img_width - 1
    )
    out[:, 3] = np.clip(
        (out[:, 3] - pad[0]) / gain, 0, img_height - 1
    )
    return out


def convert_to_original_coordinates_pose_with_params(
    detections: np.ndarray,
    pad: Tuple[int, int],
    gain: float,
    img_width: int,
    img_height: int,
    num_keypoints: int = 17,
) -> np.ndarray:
    """Convert pose detections (bbox + keypoints) to original image coordinates."""
    if len(detections) == 0:
        return detections
    out = detections.copy()
    out[:, 0] = np.clip(
        (out[:, 0] - pad[1]) / gain, 0, img_width - 1
    )
    out[:, 1] = np.clip(
        (out[:, 1] - pad[0]) / gain, 0, img_height - 1
    )
    out[:, 2] = np.clip(
        (out[:, 2] - pad[1]) / gain, 0, img_width - 1
    )
    out[:, 3] = np.clip(
        (out[:, 3] - pad[0]) / gain, 0, img_height - 1
    )
    for i in range(num_keypoints):
        out[:, 6 + i * 3] = np.clip(
            (out[:, 6 + i * 3] - pad[1]) / gain, 0, img_width - 1
        )
        out[:, 6 + i * 3 + 1] = np.clip(
            (out[:, 6 + i * 3 + 1] - pad[0]) / gain, 0, img_height - 1
        )
    return out


def convert_to_original_coordinates_seg_with_params(
    detections: np.ndarray,
    masks: np.ndarray,
    pad: Tuple[int, int],
    gain: float,
    img_width: int,
    img_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert segmentation detections and masks to original image coordinates."""
    if len(detections) == 0:
        return detections, masks
    out_det = detections.copy()
    out_det[:, 0] = np.clip(
        (out_det[:, 0] - pad[1]) / gain, 0, img_width - 1
    )
    out_det[:, 1] = np.clip(
        (out_det[:, 1] - pad[0]) / gain, 0, img_height - 1
    )
    out_det[:, 2] = np.clip(
        (out_det[:, 2] - pad[1]) / gain, 0, img_width - 1
    )
    out_det[:, 3] = np.clip(
        (out_det[:, 3] - pad[0]) / gain, 0, img_height - 1
    )
    if len(masks) == 0:
        return out_det, masks
    top, left = int(pad[0]), int(pad[1])
    unpad_h = int(img_height * gain)
    unpad_w = int(img_width * gain)
    n, mh, mw = masks.shape
    y_end = min(top + unpad_h, mh)
    x_end = min(left + unpad_w, mw)
    masks_sliced = masks[:, top:y_end, left:x_end]
    new_masks = np.zeros(
        (n, img_height, img_width), dtype=np.float32
    )
    for i, mask in enumerate(masks_sliced):
        new_masks[i] = cv2.resize(
            mask,
            (img_width, img_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return out_det, new_masks


def stream_inference_multi(
    source,
    model1: YOLOv26,
    model2: Union[YOLOv26, YOLOv26Pose],
    model3: YOLOv26Seg,
    display: bool = True,
    panel4_image_path: Optional[str] = "image.jpg",
) -> None:
    """Run YOLOv26 (det) + YOLOv26Pose + YOLOv26Seg on one input, 4-panel output.
    Each panel updates as soon as its model result is ready (faster perceived display).
    """
    # Load panel 4 image once (blank if not found)
    panel4_image: Optional[np.ndarray] = None
    if panel4_image_path and os.path.isfile(panel4_image_path):
        panel4_image = cv2.imread(panel4_image_path)
        if panel4_image is not None:
            print(f"[INFO] Panel 4 image loaded: {panel4_image_path}")
        else:
            print(f"[WARNING] Failed to load panel 4 image: {panel4_image_path}")
    elif panel4_image_path:
        print(f"[WARNING] Panel 4 image not found: {panel4_image_path}, using blank.")

    # Max frames waiting at input: excess frames are dropped (realtime backpressure)
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

    # Single input queue: main puts (frame_id, frame_bgr, meta)
    input_image_queue: "queue.Queue[tuple]" = queue.Queue()
    # Per-model queues
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

    def dispatcher_worker() -> None:
        """Copy each frame to all three model input queues."""
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

    def render_worker() -> None:
        """Update each panel independently as soon as its model result is ready."""
        win_name = "YOLOv26 Multi Faster (1:Det | 2:Pose | 3:Seg | 4:Empty)"
        window_created = False
        ref_h: Optional[int] = None
        ref_w: Optional[int] = None
        combined: Optional[np.ndarray] = None
        p4_static: Optional[np.ndarray] = None
        BLANK_COLOR = (40, 40, 40)

        def ensure_combined_and_window() -> None:
            nonlocal combined, p4_static, window_created
            if combined is not None:
                return
            gray = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
            gray[:] = BLANK_COLOR
            if panel4_image is not None:
                p4_static = cv2.resize(
                    panel4_image, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                p4_static = np.zeros((ref_h, ref_w, 3), dtype=np.uint8)
                p4_static[:] = BLANK_COLOR
            top_row = np.hstack((gray, gray))
            bottom_row = np.hstack((gray, p4_static.copy()))
            combined = np.vstack((top_row, bottom_row))
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
            window_created = True

        def get_nonblock(q: "queue.Queue") -> Optional[Any]:
            if q.empty():
                return None
            try:
                return q.get_nowait()
            except queue.Empty:
                return None

        def process_d1(d1: Any) -> bool:
            nonlocal ref_h, ref_w, combined
            if d1 is SENTINEL or stop_event.is_set():
                return True
            _fid, frame_bgr, det = d1
            if ref_h is None or ref_w is None:
                ref_h, ref_w = frame_bgr.shape[:2]
                ensure_combined_and_window()
            t0 = time.perf_counter()
            p1 = frame_bgr.copy()
            model1.draw_detections(p1, det)
            if p1.shape[:2] != (ref_h, ref_w):
                p1 = cv2.resize(p1, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            _draw_label(p1, "YOLO26s Det", (12, 32), (200, 255, 200), (25, 55, 25))
            combined[0:ref_h, 0:ref_w] = p1
            with metrics_lock:
                metrics["sum_render"] += time.perf_counter() - t0
            return False

        def process_d2(d2: Any) -> bool:
            nonlocal ref_h, ref_w, combined
            if d2 is SENTINEL or stop_event.is_set():
                return True
            _fid, frame_bgr, det = d2
            if ref_h is None or ref_w is None:
                ref_h, ref_w = frame_bgr.shape[:2]
                ensure_combined_and_window()
            t0 = time.perf_counter()
            p2 = frame_bgr.copy()
            model2.draw_detections(p2, det)
            if p2.shape[:2] != (ref_h, ref_w):
                p2 = cv2.resize(p2, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            _draw_label(p2, "YOLO26s Pose", (12, 32), (200, 235, 255), (25, 45, 55))
            combined[0:ref_h, ref_w : 2 * ref_w] = p2
            with metrics_lock:
                metrics["sum_render"] += time.perf_counter() - t0
            return False

        def process_d3(d3: Any) -> bool:
            nonlocal ref_h, ref_w, combined
            if d3 is SENTINEL or stop_event.is_set():
                return True
            _fid, frame_bgr, det, masks = d3
            if ref_h is None or ref_w is None:
                ref_h, ref_w = frame_bgr.shape[:2]
                ensure_combined_and_window()
            t0 = time.perf_counter()
            p3 = frame_bgr.copy()
            model3.draw_detections(p3, det, masks)
            if p3.shape[:2] != (ref_h, ref_w):
                p3 = cv2.resize(p3, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
            _draw_label(p3, "YOLO26s Seg", (12, 32), (255, 225, 200), (55, 35, 25))
            combined[ref_h : 2 * ref_h, 0:ref_w] = p3
            with metrics_lock:
                metrics["sum_render"] += time.perf_counter() - t0
            return False

        while True:
            d1 = get_nonblock(detections_queue_1)
            d2 = get_nonblock(detections_queue_2)
            d3 = get_nonblock(detections_queue_3)

            if d1 is None and d2 is None and d3 is None:
                try:
                    item = detections_queue_1.get(timeout=0.05)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    key = cv2.waitKey(1) & 0xFF
                    if window_created and combined is not None:
                        cv2.imshow(win_name, combined)
                    if key == ord("q") or key == 27:
                        set_stop_event()
                    continue
                d1 = item

            stop_requested = False
            if d1 is not None:
                stop_requested = process_d1(d1) or stop_requested
            if d2 is not None:
                stop_requested = process_d2(d2) or stop_requested
            if d3 is not None:
                stop_requested = process_d3(d3) or stop_requested

            if stop_requested:
                break

            if combined is not None:
                cv2.imshow(win_name, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                set_stop_event()
                break

    threads = [
        threading.Thread(target=dispatcher_worker, daemon=True),
        threading.Thread(
            target=preprocess_worker,
            args=(model1, input_queue_1, req_id_queue_1),
            daemon=True,
        ),
        threading.Thread(
            target=preprocess_worker,
            args=(model2, input_queue_2, req_id_queue_2),
            daemon=True,
        ),
        threading.Thread(
            target=preprocess_worker,
            args=(model3, input_queue_3, req_id_queue_3),
            daemon=True,
        ),
        threading.Thread(
            target=wait_worker,
            args=(model1, req_id_queue_1, output_queue_1),
            daemon=True,
        ),
        threading.Thread(
            target=wait_worker,
            args=(model2, req_id_queue_2, output_queue_2),
            daemon=True,
        ),
        threading.Thread(
            target=wait_worker,
            args=(model3, req_id_queue_3, output_queue_3),
            daemon=True,
        ),
        threading.Thread(
            target=postprocess_worker,
            args=(model1, output_queue_1, detections_queue_1),
            daemon=True,
        ),
        threading.Thread(
            target=postprocess_worker,
            args=(model2, output_queue_2, detections_queue_2),
            daemon=True,
        ),
        threading.Thread(
            target=postprocess_worker,
            args=(model3, output_queue_3, detections_queue_3),
            daemon=True,
        ),
    ]
    if display:
        threads.append(threading.Thread(target=render_worker, daemon=True))

    for t in threads:
        t.start()

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open input source: {source}")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 15)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"\n[INFO] Camera index: {source}")
    elif isinstance(source, str) and source.startswith("rtsp://"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"\n[INFO] RTSP URL: {source}")
    else:
        print(f"\n[INFO] Video file: {source}")
    print(f"[INFO] Input source resolution (WxH): {width}x{height}")
    if total_frames > 0:
        print(f"[INFO] Total frames: {total_frames}")
    if fps > 0:
        print(f"[INFO] Input source FPS: {fps:.2f}")
    print("\n[INFO] Starting inference (faster display: each panel updates independently)...")

    try:
        cnt = 0
        start_time = time.perf_counter()
        while not stop_event.is_set():
            t0 = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t1 = time.perf_counter()
            meta = {"t_read": t1 - t0}
            if input_image_queue.qsize() < MAX_PENDING_INPUT_FRAMES:
                cnt += 1
                input_image_queue.put((cnt - 1, frame_bgr, meta))

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
        set_stop_event()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        set_stop_event()
    finally:
        if not stop_event.is_set():
            input_image_queue.put(SENTINEL)
        for t in threads:
            t.join()
        if metrics["infer_completed"] == 0:
            print("[WARNING] No frames were processed.")
        else:
            elapsed = time.perf_counter() - start_time
            print_async_performance_summary(metrics, cnt, elapsed, display)
        cap.release()
        cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv26 multi (faster display): each panel updates as soon as its model result is ready."
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Object detection DXNN model path (YOLOv26).",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Pose estimation DXNN model path (YOLOv26Pose).",
    )
    parser.add_argument(
        "--model3",
        type=str,
        required=True,
        help="Instance segmentation DXNN model path (YOLOv26Seg).",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to input video.")
    group.add_argument(
        "--camera",
        type=int,
        help="Camera device index (e.g., 0 for default camera).",
    )
    group.add_argument(
        "--rtsp",
        type=str,
        help="RTSP stream URL (e.g., rtsp://ip:port/stream).",
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Do not display output window.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="image.jpg",
        help="Path to image for panel 4 (default: image.jpg). Omit or use empty to show blank.",
    )
    parser.set_defaults(display=True)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    config = Configuration()
    if version.parse(config.get_version()) < version.parse("3.0.0"):
        print(
            "[ERROR] DX-RT v3.0.0 or higher is required. "
            "Please update DX-RT to the latest version."
        )
        exit(1)

    args = parse_arguments()

    if not os.path.exists(args.model1):
        print(
            "[ERROR] --model1 .dxnn file does not exist. "
            "Please input correct model path."
        )
        exit(1)
    if not os.path.exists(args.model2):
        print(
            "[ERROR] --model2 .dxnn file does not exist (YOLOv26Pose). "
            "Please input correct model path."
        )
        exit(1)
    if not os.path.exists(args.model3):
        print(
            "[ERROR] --model3 .dxnn file does not exist (YOLOv26Seg). "
            "Please input correct model path."
        )
        exit(1)
    if args.video and not os.path.exists(args.video):
        print("[ERROR] Video file does not exist. Please input correct video path.")
        exit(1)

    model1 = YOLOv26(args.model1)
    model2 = YOLOv26Pose(args.model2)
    model3 = YOLOv26Seg(args.model3)

    source = args.video or args.camera if args.camera is not None else args.rtsp
    panel4_path = args.image.strip() or None
    stream_inference_multi(
        source, model1, model2, model3,
        display=args.display,
        panel4_image_path=panel4_path,
    )
