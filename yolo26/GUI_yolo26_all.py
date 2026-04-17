"""
PyQt5: 2×2 grid — GStreamer V4L2 camera (default) or video file (--video).
- Panel 1: YOLOv26 OD (default yolo26s-1.dxnn).
- Panel 2: YOLOv26Pose (default yolo26s-pose.dxnn).
- Panel 3: YOLOv26Seg (default yolo26s-seg.dxnn).
- Panel 4: YOLOv26Cls classification (default yolo26s-cls.dxnn), top-3 labels overlaid on frame.

Four worker threads + latest-frame queues; capture thread never waits on inference.

Each panel shows output FPS (frames displayed in the last second) in the header bar, updated once per second.

Starts in fullscreen (no window title bar). Press Q or Esc to quit.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from dx_engine import Configuration
from packaging import version
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import os
from pathlib import Path
def notify_launcher_ready() -> None:
    path = os.environ.get("DX_LAUNCHER_READY_FILE")
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # 한 줄 상태만 남기고 싶으면:
        p.write_text("ready\n", encoding="utf-8")
        # 또는 이미 런처가 만든 파일이 있으면:
        # with p.open("a", encoding="utf-8") as f:
        #     f.write("ready\n")
    except OSError:
        pass


_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", ".."))

from yolov26_multi_faster import (
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
    sys.path.insert(0, os.path.join(_ROOT, ".."))
    from yolov26pose_async import YOLOv26Pose

try:
    from instance_segmentation.yolov26seg.yolov26seg_async import YOLOv26Seg
except ImportError:
    from yolov26seg_async import YOLOv26Seg

try:
    from classification.yolov26cls.yolov26cls_sync import YOLOv26Cls
except ImportError:
    from yolov26cls_sync import YOLOv26Cls


def build_gstreamer_pipeline(
    device: str,
    width: int,
    height: int,
    fps: int,
    decoder: str,
) -> str:
    return (
        f"v4l2src device={device} ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"{decoder} ! videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


_PANEL_TITLES = (
    "YOLO26-S Object Detection",
    "YOLO26-S Pose Estimation",
    "YOLO26-S Instance Segmentation",
    "YOLO26-S Classification",
)

_PANEL_FRAME_QSS = """
QFrame#yolo_panel {
    background-color: #13151a;
    border: 1px solid #3a404c;
    border-radius: 8px;
}
"""

_PANEL_TITLE_QSS = """
QLabel#yolo_panel_title {
    color: #eceef4;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.02em;
    padding: 8px 12px 6px 12px;
    background-color: #1e222b;
    border-bottom: 1px solid #353b48;
    border-top-left-radius: 7px;
}
"""

_PANEL_FPS_QSS = """
QLabel#yolo_panel_fps {
    color: #9aa3b2;
    font-size: 12px;
    font-weight: 500;
    font-family: monospace;
    padding: 8px 12px 6px 8px;
    background-color: #1e222b;
    border-bottom: 1px solid #353b48;
    border-top-right-radius: 7px;
}
"""


def _make_titled_panel(title: str) -> Tuple[QFrame, QLabel, QLabel]:
    """One-time UI: title row (FPS top-right) + video QLabel (pixmap target only)."""
    frame = QFrame()
    frame.setObjectName("yolo_panel")
    frame.setStyleSheet(_PANEL_FRAME_QSS)
    v = QVBoxLayout(frame)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(0)
    head = QHBoxLayout()
    head.setContentsMargins(0, 0, 0, 0)
    head.setSpacing(0)
    tl = QLabel(title)
    tl.setObjectName("yolo_panel_title")
    tl.setStyleSheet(_PANEL_TITLE_QSS)
    tl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    tl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    tl.setFocusPolicy(Qt.NoFocus)
    fps_lbl = QLabel("—")
    fps_lbl.setObjectName("yolo_panel_fps")
    fps_lbl.setStyleSheet(_PANEL_FPS_QSS)
    fps_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    fps_lbl.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
    fps_lbl.setFocusPolicy(Qt.NoFocus)
    head.addWidget(tl, stretch=1)
    head.addWidget(fps_lbl, stretch=0)
    v.addLayout(head)
    img = QLabel()
    img.setAlignment(Qt.AlignCenter)
    img.setScaledContents(True)
    img.setMinimumSize(280, 160)
    img.setStyleSheet("background-color: #0c0d10; border-bottom-left-radius: 7px; border-bottom-right-radius: 7px;")
    img.setFocusPolicy(Qt.NoFocus)
    v.addWidget(img, stretch=1)
    return frame, img, fps_lbl


def bgr_numpy_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    qi = QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qi.copy())


def top3_classifications(
    model: YOLOv26Cls, output_tensors: List[np.ndarray], k: int = 3
) -> List[Tuple[str, float]]:
    output = output_tensors[0]
    logits = np.asarray(output, dtype=np.float64).reshape(-1)
    logits = logits - np.max(logits)
    expv = np.exp(np.clip(logits, -80.0, 80.0))
    probs = expv / (expv.sum() + 1e-12)
    n = probs.shape[0]
    k = min(k, n)
    idx = np.argpartition(-probs, k - 1)[:k]
    idx = idx[np.argsort(-probs[idx])]
    return [(model.classes[int(i)], float(probs[i])) for i in idx]


def draw_top3_on_bgr(vis: np.ndarray, entries: List[Tuple[str, float]]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    y0 = 36
    for rank, (name, conf) in enumerate(entries, start=1):
        short = name.split(",")[0].strip()
        if len(short) > 48:
            short = short[:45] + "..."
        line = f"{rank}. {short}  ({conf:.3f})"
        y = y0 + (rank - 1) * 32
        cv2.putText(vis, line, (12, y), font, 0.65, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(vis, line, (12, y), font, 0.65, (180, 255, 180), 2, cv2.LINE_AA)


class MediaCaptureThread(QThread):
    """BGR frames from a GStreamer pipeline (camera) or an on-disk video file."""

    frame_ready = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(
        self,
        *,
        gstreamer_pipeline: Optional[str] = None,
        video_path: Optional[str] = None,
        loop_video: bool = True,
    ) -> None:
        super().__init__()
        if (gstreamer_pipeline is None) == (video_path is None):
            raise ValueError("Set exactly one of gstreamer_pipeline or video_path")
        self._gstreamer_pipeline = gstreamer_pipeline
        self._video_path = video_path
        self._loop_video = loop_video
        self._run_flag = True
        self._cap: Optional[cv2.VideoCapture] = None

    def run(self) -> None:
        if self._video_path is not None:
            self._run_file_capture()
        else:
            #self._run_gstreamer_capture()
            self._run_v4l2_capture()

    def _run_v4l2_capture(self) -> None: 
        self._cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self.error.emit(
                "VideoCapture failed!"
            )
            return
        else:
            print(f"res: {self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        while self._run_flag:
            ok, frame = self._cap.read()
            if not ok:
                self.error.emit("cap.read() failed or stream ended.")
                break
            if self._run_flag:
                frame = self._ensure_bgr3(frame)
                if frame is not None:
                    self.frame_ready.emit(frame)
        if self._cap is not None:
            self._cap.release()
            self._cap = None


    def _run_gstreamer_capture(self) -> None:
        assert self._gstreamer_pipeline is not None
        self._cap = cv2.VideoCapture(self._gstreamer_pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            self.error.emit(
                "VideoCapture failed. Check pipeline, device, and OpenCV GStreamer build "
                "(CAP_GSTREAMER)."
            )
            return
        while self._run_flag:
            ok, frame = self._cap.read()
            if not ok:
                self.error.emit("cap.read() failed or stream ended.")
                break
            if self._run_flag:
                frame = self._ensure_bgr3(frame)
                if frame is not None:
                    self.frame_ready.emit(frame)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _run_file_capture(self) -> None:
        assert self._video_path is not None
        self._cap = cv2.VideoCapture(self._video_path)
        if not self._cap.isOpened():
            self.error.emit(f"Could not open video file: {self._video_path}")
            return
        file_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if file_fps <= 1e-3:
            file_fps = 30.0
        frame_delay = 1.0 / file_fps

        while self._run_flag:
            t_loop = time.perf_counter()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if not self._loop_video:
                    if self._run_flag:
                        self.error.emit("Video ended.")
                    break
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(self._video_path)
                    if not self._cap.isOpened():
                        self.error.emit("Video loop failed to reopen file.")
                        break
                    ok, frame = self._cap.read()
                if not ok or frame is None:
                    self.error.emit("Video file has no readable frames.")
                    break
            if self._run_flag:
                frame = self._ensure_bgr3(frame)
                if frame is not None:
                    self.frame_ready.emit(frame)
            elapsed = time.perf_counter() - t_loop
            to_sleep = frame_delay - elapsed
            if to_sleep > 0 and self._run_flag:
                time.sleep(to_sleep)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @staticmethod
    def _ensure_bgr3(frame: np.ndarray) -> Optional[np.ndarray]:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.shape[2] == 3:
            return frame
        return None

    def stop(self) -> None:
        self._run_flag = False


def _put_latest_frame(q: "queue.Queue", bgr: np.ndarray) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(bgr)
    except queue.Full:
        pass


class ODWorkerThread(QThread):
    od_frame_ready = pyqtSignal(object)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model_path = model_path
        self._frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._run_flag = True
        self._model: Optional[YOLOv26] = None

    def enqueue_frame(self, bgr: np.ndarray) -> None:
        if not self._run_flag:
            return
        _put_latest_frame(self._frame_q, bgr.copy())

    def stop(self) -> None:
        self._run_flag = False

    def run(self) -> None:
        self._model = YOLOv26(self._model_path)
        model = self._model
        assert model is not None

        while self._run_flag:
            try:
                bgr = self._frame_q.get(timeout=0.05)
            except queue.Empty:
                continue
            while self._run_flag:
                try:
                    bgr = self._frame_q.get_nowait()
                except queue.Empty:
                    break
            if not self._run_flag:
                break

            input_tensor = model.preprocess(bgr)
            req_id = model.ie.run_async([input_tensor])
            output_tensors = model.ie.wait(req_id)
            detections = model.postprocess(output_tensors)
            detections = convert_to_original_coordinates_with_params(
                detections,
                model.pad,
                model.gain,
                model.img_width,
                model.img_height,
            )
            vis = bgr.copy()
            model.draw_detections(vis, detections)
            if self._run_flag:
                self.od_frame_ready.emit(vis)


class PoseWorkerThread(QThread):
    pose_frame_ready = pyqtSignal(object)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model_path = model_path
        self._frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._run_flag = True
        self._model: Optional[YOLOv26Pose] = None

    def enqueue_frame(self, bgr: np.ndarray) -> None:
        if not self._run_flag:
            return
        _put_latest_frame(self._frame_q, bgr.copy())

    def stop(self) -> None:
        self._run_flag = False

    def run(self) -> None:
        self._model = YOLOv26Pose(self._model_path)
        model = self._model
        assert model is not None

        while self._run_flag:
            try:
                bgr = self._frame_q.get(timeout=0.05)
            except queue.Empty:
                continue
            while self._run_flag:
                try:
                    bgr = self._frame_q.get_nowait()
                except queue.Empty:
                    break
            if not self._run_flag:
                break

            input_tensor = model.preprocess(bgr)
            req_id = model.ie.run_async([input_tensor])
            output_tensors = model.ie.wait(req_id)
            detections = model.postprocess(output_tensors)
            detections = convert_to_original_coordinates_pose_with_params(
                detections,
                model.pad,
                model.gain,
                model.img_width,
                model.img_height,
                model.num_keypoints,
            )
            vis = bgr.copy()
            model.draw_detections(vis, detections)
            if self._run_flag:
                self.pose_frame_ready.emit(vis)


class SegWorkerThread(QThread):
    seg_frame_ready = pyqtSignal(object)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model_path = model_path
        self._frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._run_flag = True
        self._model: Optional[YOLOv26Seg] = None

    def enqueue_frame(self, bgr: np.ndarray) -> None:
        if not self._run_flag:
            return
        _put_latest_frame(self._frame_q, bgr.copy())

    def stop(self) -> None:
        self._run_flag = False

    def run(self) -> None:
        self._model = YOLOv26Seg(self._model_path)
        model = self._model
        assert model is not None

        while self._run_flag:
            try:
                bgr = self._frame_q.get(timeout=0.05)
            except queue.Empty:
                continue
            while self._run_flag:
                try:
                    bgr = self._frame_q.get_nowait()
                except queue.Empty:
                    break
            if not self._run_flag:
                break

            input_tensor = model.preprocess(bgr)
            req_id = model.ie.run_async([input_tensor])
            output_tensors = model.ie.wait(req_id)
            detections, masks = model.postprocess(output_tensors)
            detections, masks = convert_to_original_coordinates_seg_with_params(
                detections,
                masks,
                model.pad,
                model.gain,
                model.img_width,
                model.img_height,
            )
            vis = bgr.copy()
            if len(detections) > 0 and len(masks) > 0:
                model.draw_detections(vis, detections, masks)
            if self._run_flag:
                self.seg_frame_ready.emit(vis)


class ClsWorkerThread(QThread):
    cls_frame_ready = pyqtSignal(object)

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model_path = model_path
        self._frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self._run_flag = True
        self._model: Optional[YOLOv26Cls] = None

    def enqueue_frame(self, bgr: np.ndarray) -> None:
        if not self._run_flag:
            return
        _put_latest_frame(self._frame_q, bgr.copy())

    def stop(self) -> None:
        self._run_flag = False

    def run(self) -> None:
        self._model = YOLOv26Cls(self._model_path)
        model = self._model
        assert model is not None

        while self._run_flag:
            try:
                bgr = self._frame_q.get(timeout=0.05)
            except queue.Empty:
                continue
            while self._run_flag:
                try:
                    bgr = self._frame_q.get_nowait()
                except queue.Empty:
                    break
            if not self._run_flag:
                break

            input_tensor = model.preprocess(bgr)
            req_id = model.ie.run_async([input_tensor])
            output_tensors = model.ie.wait(req_id)
            top3 = top3_classifications(model, output_tensors, k=3)
            vis = bgr.copy()
            draw_top3_on_bgr(vis, top3)
            if self._run_flag:
                self.cls_frame_ready.emit(vis)


class QuadViewODPoseSegClsWindow(QMainWindow):
    def __init__(
        self,
        od_model_path: str,
        pose_model_path: str,
        seg_model_path: str,
        cls_model_path: str,
        *,
        gstreamer_pipeline: Optional[str] = None,
        video_path: Optional[str] = None,
        loop_video: bool = True,
    ) -> None:
        super().__init__()
        self.setWindowTitle("GUI_yolo26_all")
        self._labels: list[QLabel] = []
        self._fps_labels: list[QLabel] = []
        self._fps_counts = [0, 0, 0, 0]
        central = QWidget()
        central.setFocusPolicy(Qt.StrongFocus)
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setSpacing(8)
        grid.setContentsMargins(8, 8, 8, 8)

        cells = ((0, 0), (0, 1), (1, 0), (1, 1))
        for i, title in enumerate(_PANEL_TITLES):
            frame, img, fps_lbl = _make_titled_panel(title)
            grid.addWidget(frame, cells[i][0], cells[i][1])
            self._labels.append(img)
            self._fps_labels.append(fps_lbl)

        self._fps_timer = QTimer(self)
        self._fps_timer.setInterval(1000)
        self._fps_timer.timeout.connect(self._tick_fps_display)
        self._fps_timer.start()

        self._od = ODWorkerThread(od_model_path)
        self._od.od_frame_ready.connect(self._on_od_frame)
        self._od.start()

        self._pose = PoseWorkerThread(pose_model_path)
        self._pose.pose_frame_ready.connect(self._on_pose_frame)
        self._pose.start()

        self._seg = SegWorkerThread(seg_model_path)
        self._seg.seg_frame_ready.connect(self._on_seg_frame)
        self._seg.start()

        self._cls = ClsWorkerThread(cls_model_path)
        self._cls.cls_frame_ready.connect(self._on_cls_frame)
        self._cls.start()

        self._cap_thread = MediaCaptureThread(
            gstreamer_pipeline=gstreamer_pipeline,
            video_path=video_path,
            loop_video=loop_video,
        )
        self._cap_thread.frame_ready.connect(self._on_camera_frame)
        self._cap_thread.error.connect(self._on_error)
        self._cap_thread.start()

        for key in (Qt.Key_Escape, Qt.Key_Q):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(self.close)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        cw = self.centralWidget()
        if cw is not None:
            cw.setFocus(Qt.OtherFocusReason)

    def _tick_fps_display(self) -> None:
        for i, lbl in enumerate(self._fps_labels):
            n = self._fps_counts[i]
            self._fps_counts[i] = 0
            lbl.setText(f"{n} FPS" if n else "—")

    def _on_camera_frame(self, bgr: np.ndarray) -> None:
        self._od.enqueue_frame(bgr)
        self._pose.enqueue_frame(bgr)
        self._seg.enqueue_frame(bgr)
        self._cls.enqueue_frame(bgr)

    def _on_od_frame(self, bgr: np.ndarray) -> None:
        self._fps_counts[0] += 1
        self._labels[0].setPixmap(bgr_numpy_to_qpixmap(bgr))

    def _on_pose_frame(self, bgr: np.ndarray) -> None:
        self._fps_counts[1] += 1
        self._labels[1].setPixmap(bgr_numpy_to_qpixmap(bgr))

    def _on_seg_frame(self, bgr: np.ndarray) -> None:
        self._fps_counts[2] += 1
        self._labels[2].setPixmap(bgr_numpy_to_qpixmap(bgr))

    def _on_cls_frame(self, bgr: np.ndarray) -> None:
        self._fps_counts[3] += 1
        self._labels[3].setPixmap(bgr_numpy_to_qpixmap(bgr))

    def _on_error(self, msg: str) -> None:
        for lab in self._labels:
            lab.setText(msg)
        print(f"[ERROR] {msg}", file=sys.stderr)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._fps_timer.stop()
        self._od.stop()
        self._pose.stop()
        self._seg.stop()
        self._cls.stop()
        self._cap_thread.stop()
        self._cap_thread.wait(3000)
        self._od.wait(15000)
        self._pose.wait(15000)
        self._seg.wait(20000)
        self._cls.wait(15000)
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "PyQt5 2×2: OD | Pose | Seg | classification top-3. "
            "Default input: GStreamer V4L2 camera; use --video for a file."
        )
    )
    p.add_argument("--model", type=str, default="models/yolo26s-1.dxnn", help="YOLOv26 detection .dxnn")
    p.add_argument(
        "--model-pose", type=str, default="models/yolo26s-pose.dxnn", help="YOLOv26Pose .dxnn"
    )
    p.add_argument("--model-seg", type=str, default="models/yolo26s-seg.dxnn", help="YOLOv26Seg .dxnn")
    p.add_argument(
        "--model-cls",
        type=str,
        default="models/yolo26s-cls.dxnn",
        help="YOLOv26Cls .dxnn (default: yolo26s-cls.dxnn)",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        metavar="PATH",
        help="Video file path as input (OpenCV). Default is V4L2 camera via GStreamer.",
    )
    p.add_argument(
        "--no-loop-video",
        action="store_true",
        help="With --video, exit when the file ends instead of looping.",
    )
    p.add_argument("--device", default="/dev/video0", help="V4L2 device (ignored with --video)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--decoder",
        default="mppjpegdec",
        choices=("mppjpegdec", "jpegdec"),
    )
    return p.parse_args()


def main() -> None:
    config = Configuration()
    if version.parse(config.get_version()) < version.parse("3.0.0"):
        print("[ERROR] DX-RT v3.0.0 or higher is required.")
        sys.exit(1)

    args = parse_args()
    paths = {
        "OD": os.path.abspath(args.model),
        "Pose": os.path.abspath(args.model_pose),
        "Seg": os.path.abspath(args.model_seg),
        "Cls": os.path.abspath(args.model_cls),
    }
    for name, path in paths.items():
        if not os.path.isfile(path):
            print(f"[ERROR] {name} model not found: {path}")
            sys.exit(1)

    app = QApplication(sys.argv)
    if args.video:
        video_abs = os.path.abspath(args.video)
        if not os.path.isfile(video_abs):
            print(f"[ERROR] Video file not found: {video_abs}")
            sys.exit(1)
        print(f"[INFO] Input: video file {video_abs}")
        if args.no_loop_video:
            print("[INFO] Video will not loop (stop at EOF)")
        win = QuadViewODPoseSegClsWindow(
            paths["OD"],
            paths["Pose"],
            paths["Seg"],
            paths["Cls"],
            video_path=video_abs,
            loop_video=not args.no_loop_video,
        )
    else:
        pipeline = build_gstreamer_pipeline(
            args.device, args.width, args.height, args.fps, args.decoder
        )
        print(f"[INFO] Input: GStreamer camera\n  {pipeline}")
        win = QuadViewODPoseSegClsWindow(
            paths["OD"],
            paths["Pose"],
            paths["Seg"],
            paths["Cls"],
            gstreamer_pipeline=pipeline,
        )

    for name, path in paths.items():
        print(f"[INFO] {name} model: {path}")
    notify_launcher_ready()
    win.showFullScreen()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
