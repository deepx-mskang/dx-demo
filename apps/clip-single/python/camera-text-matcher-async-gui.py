#!/usr/bin/env python3
"""
Real-time Camera Text Matcher GUI (PyQt5) using CLIP + DXNN Async Image Encoder

Left: live camera preview
Right: text list with real-time similarity bars and history
"""

import os
import sys
import argparse
import hashlib
import html
import json
import re
import queue
import threading
import time

import cv2
import numpy as np
import torch
import onnxruntime
import open_clip
from PIL import Image
from torchvision import transforms

from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QVBoxLayout, QHBoxLayout,
    QProgressBar, QScrollArea, QPushButton,
    QFrame, QGridLayout, QSizePolicy, QShortcut,
    QPlainTextEdit,
)

from dx_engine import InferenceEngine

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


class ONNXModel(torch.nn.Module):
    """Thin torch.nn.Module wrapper around onnxruntime for CLIP text encoder."""

    def __init__(self, model_path: str):
        super().__init__()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")
        self.model = onnxruntime.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
                )
        self.output_names = [x.name for x in self.model.get_outputs()]

    def forward(self, x):
        """Run ONNX session; supports single tensor or list/tuple for multi-input models."""
        onnx_inputs = self.model.get_inputs()
        inputs_dict = {}
        if len(onnx_inputs) > 1 and isinstance(x, (tuple, list)):
            for i, input_node in enumerate(onnx_inputs):
                if i < len(x):
                    inputs_dict[input_node.name] = x[i].cpu().numpy()
        else:
            inputs_dict[onnx_inputs[0].name] = x.cpu().numpy()

        pred = self.model.run(self.output_names, inputs_dict)

        if isinstance(pred, list):
            pred = pred[0] if len(pred) == 1 else np.stack(pred)
        return torch.from_numpy(pred) if isinstance(pred, np.ndarray) else torch.Tensor(pred)


class TextEncoder(torch.nn.Module):
    """CLIP text tower implemented as ONNX inference."""

    def __init__(self, onnx_path: str):
        super().__init__()
        self.text_encoder_onnx = ONNXModel(onnx_path)

    def forward(self, text_tokens):
        """text_tokens: CLIP tokenizer output (batch of token ids)."""
        return self.text_encoder_onnx(text_tokens)


def get_text_feature_cache_path(args) -> Path:
    """Return a stable cache path for precomputed text features."""
    cache_dir = Path(__file__).resolve().parent / '.cache' / 'text_features'
    encoder_path = Path(args.text_encoder).resolve()
    cache_payload = {
        'text_encoder_path': str(encoder_path),
        'text_encoder_size': encoder_path.stat().st_size if encoder_path.exists() else None,
        'text_encoder_mtime_ns': encoder_path.stat().st_mtime_ns if encoder_path.exists() else None,
        'model_name': args.model_name,
        'texts': list(args.texts),
        'no_normalize': bool(args.no_normalize),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, ensure_ascii=True, sort_keys=True).encode('utf-8')
    ).hexdigest()
    return cache_dir / f'{cache_key}.pt'


def load_cached_text_features(args):
    """Load cached text features if they exist and match the current inputs."""
    cache_path = get_text_feature_cache_path(args)
    if not cache_path.exists():
        return None

    try:
        payload = torch.load(cache_path, map_location='cpu')
    except Exception:
        return None

    features = payload.get('text_features')
    if not isinstance(features, torch.Tensor):
        return None
    return features


def save_cached_text_features(args, text_features: torch.Tensor):
    """Persist precomputed text features for reuse on later launches."""
    cache_path = get_text_feature_cache_path(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'text_features': text_features.detach().cpu(),
        'created_at': time.time(),
        'texts': list(args.texts),
        'model_name': args.model_name,
        'text_encoder': str(Path(args.text_encoder).resolve()),
        'normalized': not bool(args.no_normalize),
    }
    torch.save(payload, cache_path)
    return cache_path


class ImageEncoderAsync:
    """DXNN image encoder with async run + callback; results arrive on a queue."""

    def __init__(self, dxnn_path: str):
        if not os.path.isfile(dxnn_path):
            raise FileNotFoundError(f"DXNN model file not found: {dxnn_path}")
        self.engine = InferenceEngine(dxnn_path)
        self.input_info = self.engine.get_input_tensors_info()
        self.result_queue = queue.Queue()
        self._closing = threading.Event()
        self._req_lock = threading.Lock()
        self._outstanding_req_ids = set()
        self.engine.register_callback(self._inference_callback)

    def _inference_callback(self, outputs, user_arg):
        """Engine callback: push (job_id, features) to result_queue for the GUI thread."""
        if self._closing.is_set():
            return 0
        job_id, _frame_ts = user_arg
        if isinstance(outputs, list):
            image_features = outputs[0]
        else:
            image_features = outputs
        image_features = np.array(image_features)
        try:
            self.result_queue.put((job_id, image_features), block=False)
        except Exception:
            pass
        return 0

    def encode_async(self, image_array: np.ndarray, job_id: int):
        """Submit one NCHW float batch; returns request id for completion tracking."""
        if self._closing.is_set():
            return None
        req_id = self.engine.run_async([image_array], user_arg=(job_id, time.time()))
        try:
            with self._req_lock:
                self._outstanding_req_ids.add(req_id)
        except Exception:
            pass
        return req_id

    def get_result(self, timeout=None):
        """Non-blocking when timeout=0; returns (job_id, image_features) or None."""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def notify_request_completed(self, req_id):
        """Drop completed request id from the outstanding set."""
        if req_id is None:
            return
        try:
            with self._req_lock:
                self._outstanding_req_ids.discard(req_id)
        except Exception:
            pass

    def close(self, wait_timeout_sec=2.0):
        """Signal shutdown and best-effort wait/stop on the underlying engine."""
        self._closing.set()
        engine = getattr(self, 'engine', None)
        if engine is None:
            return
        # Drop native callback if the SDK supports it (avoids use-after-free on teardown).
        for name in ('unregister_callback', 'clear_callback', 'remove_callback'):
            fn = getattr(engine, name, None)
            if not callable(fn):
                continue
            try:
                fn(self._inference_callback)
            except TypeError:
                try:
                    fn()
                except Exception:
                    pass
            except Exception:
                pass
            break
        # DXNN API surface varies by build; probe common wait/sync method names.
        for name in ['wait_all', 'waitAll', 'synchronize', 'sync', 'join', 'flush', 'drain', 'wait_for_all_requests']:
            fn = getattr(engine, name, None)
            if callable(fn):
                try:
                    fn(wait_timeout_sec)
                except TypeError:
                    fn()
                except Exception:
                    pass
                break
        # Release native resources; method name depends on the DXNN SDK.
        for name in ['stop', 'close', 'shutdown', 'release', 'destroy']:
            fn = getattr(engine, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break


def create_image_transform():
    """OpenCLIP / standard ImageNet normalization for 224×224 CLIP image input."""
    return transforms.Compose([
        transforms.Resize(size=224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def preprocess_frame(frame: np.ndarray, transform):
    """BGR OpenCV frame → NCHW float tensor batch (single image) for the encoder."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_tensor = transform(pil_image)
    image_array = image_tensor.numpy()
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def crop_center(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Return a centered crop of the given size from the input frame."""
    h, w = frame.shape[:2]
    if target_width >= w or target_height >= h:
        return frame
    x = (w - target_width) // 2
    y = (h - target_height) // 2
    return frame[y:y + target_height, x:x + target_width]


def is_video_file_source(input_source) -> bool:
    """True when input_source is a path to an on-disk video file."""
    return isinstance(input_source, str) and os.path.isfile(input_source)


def describe_frame(frame: np.ndarray) -> str:
    """Compact frame diagnostics for camera/preview debugging."""
    if frame is None:
        return 'None'
    try:
        channel_mean = frame.mean(axis=(0, 1))
        mean_text = ','.join(f'{v:.1f}' for v in channel_mean)
        return (
            f'shape={frame.shape}, dtype={frame.dtype}, strides={frame.strides}, '
            f'min={int(frame.min())}, max={int(frame.max())}, mean={frame.mean():.1f}, '
            f'channel_mean=[{mean_text}], contiguous={frame.flags.c_contiguous}'
        )
    except Exception as e:
        return f'shape={getattr(frame, "shape", None)}, stats_error={e}'


def is_nearly_black_frame(frame: np.ndarray) -> bool:
    """True when the captured pixels are effectively all black."""
    if frame is None or frame.size == 0:
        return True
    try:
        return float(frame.mean()) < 1.0 and int(frame.max()) <= 3
    except Exception:
        return False


def normalize_features(features):
    """L2-normalize embedding rows for cosine similarity via dot product."""
    if isinstance(features, torch.Tensor):
        return features / features.norm(dim=-1, keepdim=True)
    else:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / norms


def compute_similarity(text_features, image_features):
    """Dot product between (possibly batched) text and image feature vectors."""
    if isinstance(text_features, torch.Tensor):
        text_features = text_features.cpu().numpy()
    if isinstance(image_features, torch.Tensor):
        image_features = image_features.cpu().numpy()
    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)
    similarity = np.dot(text_features, image_features.T)
    return similarity.squeeze()


# Minimum similarity to participate in best / second-best UI highlight and status "Best".
HIGHLIGHT_MIN_SCORE = 0.25
HIGHLIGHT_MAX_SCORE = 0.35

# Match TextRowWidget bar colors for the left-panel summary line
MATCH_COLOR_FIRST = '#d62828'
MATCH_COLOR_SECOND = '#f59f00'


def similarity_to_bar_value(score: float) -> int:
    """Map raw cosine similarity to 0–1000 for QProgressBar.

    Below HIGHLIGHT_MIN_SCORE grows gently;
    slopes increase through HIGHLIGHT_MIN_SCORE + 0.05; HIGHLIGHT_MAX_SCORE fills the bar
    (treat as strong match). Values above 0.3 stay at full.
    """
    s = float(score)
    if s <= 0.0:
        return 0
    if s >= HIGHLIGHT_MAX_SCORE:
        return 1000

    if s <= HIGHLIGHT_MIN_SCORE:
        # Segment 1: score 0..0.2 → bar 0..280 (gentle rise)
        t = s / HIGHLIGHT_MIN_SCORE
        return int(round(280 * t))
    if s <= HIGHLIGHT_MIN_SCORE + 0.05:
        # Segment 2: 0.2..0.25 → 280..620 (steeper)
        t = (s - HIGHLIGHT_MIN_SCORE) / 0.05
        return int(round(280 + 340 * t))
    # Segment 3: 0.25..0.3 → 620..1000 (strong match band)
    t = (s - HIGHLIGHT_MIN_SCORE + 0.05) / 0.05
    return int(round(620 + 380 * t))


class CameraThread(QThread):
    """Background thread: read camera or video file and emit BGR frames at ~target FPS."""

    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, input_source, width, height, fps):
        super().__init__()
        self.input_source = input_source
        self.width = width
        self.height = height
        self.fps = fps
        self._stop = threading.Event()
        self._frame_count = 0
        self._black_frame_count = 0
        self._read_fail_count = 0
        self._saved_black_debug_frame = False
        self._saved_nonblack_debug_frame = False

    def _resolve_capture_source(self):
        """Numeric string sources become int indices for cv2.VideoCapture."""
        if isinstance(self.input_source, str) and self.input_source.isdigit():
            return int(self.input_source)
        return self.input_source

    def _is_video_file(self):
        """True when input_source is a path to an on-disk video file."""
        return is_video_file_source(self.input_source)

    def _should_try_v4l2(self, capture_source):
        """Linux camera devices are more reliable through V4L2 than GStreamer URI probing."""
        if isinstance(capture_source, int):
            return True
        return isinstance(capture_source, str) and capture_source.startswith('/dev/video')

    def _open_capture(self, capture_source):
        """Open capture source, preferring V4L2 for Linux camera devices."""
        if self._should_try_v4l2(capture_source):
            cap = cv2.VideoCapture(capture_source, cv2.CAP_V4L2)
            print(f'Camera debug: tried CAP_V4L2, opened={cap.isOpened()}')
            if cap.isOpened():
                return cap
            cap.release()
            print('Camera debug: CAP_V4L2 failed; falling back to OpenCV default backend')
        return cv2.VideoCapture(capture_source)

    def run(self):
        """Capture loop: emit frames until stop(); falls back to a static test image if open fails."""
        capture_source = self._resolve_capture_source()
        print(
            f'Camera debug: opening input={self.input_source!r}, '
            f'resolved={capture_source!r}, requested={self.width}x{self.height}@{self.fps}'
        )
        cap = self._open_capture(capture_source)
        use_test_image = False
        is_video_file = self._is_video_file()
        frame_interval = max(0, (1.0 / max(1, self.fps)) - 0.005)

        if not cap.isOpened():
            print(f"Input {self.input_source} not available, using test image")
            use_test_image = True
            test_image = cv2.imread('../../workspace/assets/clip-single/img-encoder-sample-1.png')
            if test_image is None:
                self.frame_ready.emit(None)
                return
            test_image = cv2.resize(test_image, (self.width, self.height))
        else:
            source_kind = 'video file' if is_video_file else 'camera'
            print(f"{source_kind.capitalize()} opened successfully: {self.input_source}")
            try:
                print(f'Camera debug: OpenCV backend={cap.getBackendName()}')
            except Exception as e:
                print(f'Camera debug: backend name unavailable: {e}')
            if is_video_file:
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                if source_fps and source_fps > 0:
                    frame_interval = max(0, (1.0 / source_fps) - 0.005)
            else:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(
                    f'Camera requested {self.width}x{self.height}@{self.fps}, '
                    f'actual {actual_width:.0f}x{actual_height:.0f}@{actual_fps:.1f}'
                )

        while not self._stop.is_set():
            if use_test_image:
                frame = test_image.copy()
            else:
                ret, frame = cap.read()
                if not ret:
                    self._read_fail_count += 1
                    if self._read_fail_count <= 5 or self._read_fail_count % 30 == 0:
                        print(
                            f'Camera debug: cap.read() failed '
                            f'count={self._read_fail_count}, source={self.input_source!r}'
                        )
                    if is_video_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    continue
                if is_video_file:
                    frame = cv2.resize(frame, (self.width, self.height))
            self._frame_count += 1
            if is_nearly_black_frame(frame):
                self._black_frame_count += 1
                if not self._saved_black_debug_frame:
                    self._saved_black_debug_frame = True
                    cv2.imwrite('debug_camera_black_frame.png', frame)
                    print('Camera debug: saved black sample to debug_camera_black_frame.png')
            elif not self._saved_nonblack_debug_frame:
                self._saved_nonblack_debug_frame = True
                cv2.imwrite('debug_camera_nonblack_frame.png', frame)
                print('Camera debug: saved non-black sample to debug_camera_nonblack_frame.png')

            if self._frame_count <= 10 or self._frame_count % max(1, self.fps * 5) == 0:
                print(
                    f'Camera debug: frame #{self._frame_count}, '
                    f'black_frames={self._black_frame_count}, {describe_frame(frame)}'
                )
            self.frame_ready.emit(frame)
            time.sleep(frame_interval)

        if not use_test_image:
            cap.release()

    def stop(self):
        """Request thread exit and block until run() finishes."""
        self._stop.set()
        self.wait()


class TextRowWidget(QWidget):
    """One query string: similarity bar, numeric score, and optional top-2 highlight styling."""

    def __init__(self, text: str):
        super().__init__()
        self.text = text
        self.history = []
        self.max_history = 32

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setStyleSheet('font-size: 14px; color: #111111;')

        self.score_label = QLabel('0.000')
        self.score_label.setStyleSheet('font-size: 12px; color: #111111; min-width: 56px;')

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(16)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.addWidget(self.progress, 1)
        top_row.addWidget(self.score_label)

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        layout.addLayout(top_row)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.set_highlight_rank(None)

    def update_score(self, score: float):
        """Refresh raw score label and non-linear progress bar; append to rolling history."""
        self.history.append(score)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        value = similarity_to_bar_value(score)
        value = max(0, min(1000, value))
        self.progress.setValue(value)
        self.score_label.setText(f"{score:.3f}")

    def set_highlight_rank(self, rank):
        """rank 0 = best (red), 1 = second (amber), None = default styling."""
        chunk_color = '#4a5568'
        border_color = '#cbd5e1'
        label_color = '#111111'
        score_color = '#111111'
        font_weight = 'normal'
        bg_color = '#f8fafc'

        if rank == 0:
            chunk_color = '#d62828'
            border_color = '#ff7b7b'
            label_color = '#111111'
            score_color = '#111111'
            font_weight = '700'
            bg_color = '#fff1f1'
        elif rank == 1:
            chunk_color = '#f59f00'
            border_color = '#ffd43b'
            label_color = '#111111'
            score_color = '#111111'
            font_weight = '600'
            bg_color = '#fff9db'

        self.label.setStyleSheet(
            f'font-size: 14px; color: {label_color}; font-weight: {font_weight};'
        )
        self.score_label.setStyleSheet(
            f'font-size: 12px; color: {score_color}; min-width: 56px; font-weight: {font_weight};'
        )
        self.setStyleSheet(
            f'background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 8px;'
        )
        self.progress.setStyleSheet(
            f'''
            QProgressBar {{
                border: 1px solid {border_color};
                border-radius: 4px;
                background-color: #e5e7eb;
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                border-radius: 4px;
            }}
            '''
        )


class MainWindow(QMainWindow):
    """Main UI: live preview, async image encode, text–image similarity and ranking."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle('Camera Text Matcher GUI (Async)')
        self.resize(1920, 1080)

        container = QWidget()
        self.setCentralWidget(container)

        main_layout = QHBoxLayout()
        container.setLayout(main_layout)

        # Left column: live BGR preview + status / best-match line
        self.preview_width = 1280
        self.preview_height = 720
        self.video_label = QLabel('Camera feed not ready')
        #self.video_label.setFixedSize(960, 720)
        self.video_label.setFixedSize(self.preview_width, self.preview_height)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet('background-color: #111; color: #fff;')

        left_box = QVBoxLayout()
        left_box.addWidget(self.video_label)

        # Model/source strip vs match summary share vertical space in a 1 : 2 height ratio
        self.match_panel = QWidget()
        self.match_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        match_layout = QVBoxLayout(self.match_panel)
        match_layout.setContentsMargins(0, 0, 0, 0)
        match_layout.setSpacing(8)

        # Match summary (2x): line 1 = best, line 2 = runner-up (rich text, bar colors)
        self.status_label = QLabel('Starting up…')
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.status_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.status_label.setMinimumHeight(96)
        self.status_label.setStyleSheet(
            '''
            QLabel {
                font-size: 36px;
                font-weight: 700;
                color: #ffffff;
                background-color: #202632;
                border: 2px solid #4b5563;
                border-radius: 8px;
                padding: 14px 18px;
            }
            '''
        )
        match_layout.addWidget(self.status_label, 2)

        # Compact area (1x): input source + model / encoder names (demo context)
        self.source_info_label = QLabel()
        self.source_info_label.setWordWrap(True)
        self.source_info_label.setTextFormat(Qt.PlainText)
        self.source_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.source_info_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.source_info_label.setMinimumHeight(48)
        self.source_info_label.setStyleSheet(
            '''
            QLabel {
                font-size: 28px;
                font-weight: 500;
                color: #94a3b8;
                background-color: #1a1f2e;
                border: 1px solid #374151;
                border-radius: 6px;
                padding: 8px 12px;
            }
            '''
        )
        match_layout.addWidget(self.source_info_label, 1)

        left_box.addWidget(self.match_panel, 1)

        main_layout.addLayout(left_box)

        # Right column: scrollable list of TextRowWidget (bar + score per text)
        right_box = QVBoxLayout()

        add_text_box = QHBoxLayout()
        add_text_box.setContentsMargins(0, 0, 0, 0)
        add_text_box.setSpacing(8)

        self.add_text_editor = QPlainTextEdit()
        self.add_text_editor.setPlaceholderText('Add 1-2 lines, then Apply')
        self.add_text_editor.setFixedHeight(56)
        self.add_text_editor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.add_text_editor.setStyleSheet(
            '''
            QPlainTextEdit {
                background-color: #ffffff;
                color: #111827;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 13px;
            }
            '''
        )
        add_text_box.addWidget(self.add_text_editor, 1)

        self.apply_text_button = QPushButton('Apply')
        self.apply_text_button.setFixedHeight(56)
        self.apply_text_button.setStyleSheet(
            '''
            QPushButton {
                background-color: #0f766e;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 0 14px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:disabled {
                background-color: #94a3b8;
            }
            '''
        )
        self.apply_text_button.clicked.connect(self.apply_additional_texts)
        add_text_box.addWidget(self.apply_text_button)

        right_box.addLayout(add_text_box)

        self.text_widgets = []
        self.text_container = QWidget()
        self.text_layout = QVBoxLayout()
        self.text_layout.setContentsMargins(0, 0, 0, 0)
        self.text_layout.setSpacing(6)
        self.text_container.setLayout(self.text_layout)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.text_container)
        right_box.addWidget(self.scroll)

        main_layout.addLayout(right_box)

        self.transform = create_image_transform()
        self.text_features = None
        self.texts = list(args.texts)
        self.text_encoder = None
        self.tokenizer = None
        self._is_video_input = is_video_file_source(args.input)

        # Async pipeline: monotonic job_id maps pending encode → frame copy and request id
        self._job_id = 0
        self._frame_index = 0
        self._pending_jobs = {}
        self._pending_reqs = {}
        self._shutting_down = False
        self._preview_debug_count = 0

        self.image_encoder = None
        self.camera_thread = None

        self.setup_text_widgets()
        self.setup_models()
        self.setup_camera()
        self._refresh_source_info_label()

        # Poll DXNN results on the GUI thread (async callbacks fill the encoder queue)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start(30)

        # ESC: quit from anywhere. Q: quit unless typing in the add-text field (so "q" can be typed).
        esc_quit = QShortcut(QKeySequence(Qt.Key_Escape), self)
        esc_quit.setContext(Qt.ApplicationShortcut)
        esc_quit.activated.connect(self.close)
        q_quit = QShortcut(QKeySequence(Qt.Key_Q), self)
        q_quit.setContext(Qt.ApplicationShortcut)
        q_quit.activated.connect(self._close_on_q_unless_add_text_focused)

    def _close_on_q_unless_add_text_focused(self):
        fw = QApplication.focusWidget()
        if fw is not None and (
            fw is self.add_text_editor or self.add_text_editor.isAncestorOf(fw)
        ):
            return
        self.close()

    def _refresh_source_info_label(self):
        """One-line summary: capture source and model files (for demos)."""
        inp = str(self.args.input)
        name = self.args.model_name
        te = os.path.basename(self.args.text_encoder)
        ie = os.path.basename(self.args.image_encoder)
        self.source_info_label.setText(
            f'Input Source: {inp}  ·  Image encoder: {ie}'
        )

    def _set_status_plain(self, text: str):
        self.status_label.setTextFormat(Qt.PlainText)
        self.status_label.setText(text)

    def _set_match_status_rich(self, html: str):
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setText(html)

    def setup_text_widgets(self):
        """Build one TextRowWidget per CLI text string."""
        for txt in self.texts:
            widget = TextRowWidget(txt)
            self.text_layout.addWidget(widget)
            self.text_widgets.append(widget)

    def rebuild_text_widgets(self):
        """Recreate text widgets after the active query set changes."""
        while self.text_layout.count():
            item = self.text_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.text_widgets = []
        self.setup_text_widgets()

    def ensure_text_encoder_ready(self):
        """Load tokenizer and text encoder lazily for startup and runtime additions."""
        if self.text_encoder is None:
            self._set_status_plain('Loading text encoder…')
            self.text_encoder = TextEncoder(self.args.text_encoder)
        if self.tokenizer is None:
            self._set_status_plain('Loading tokenizer…')
            self.tokenizer = open_clip.get_tokenizer(self.args.model_name)

    def compute_text_features_for_texts(self, texts):
        """Load cached text features or encode a fresh feature matrix for the given texts."""
        previous_texts = list(self.args.texts)
        self.args.texts = list(texts)
        try:
            cached_text_features = load_cached_text_features(self.args)
            if cached_text_features is not None:
                self._set_status_plain('Loaded cached text features…')
                return cached_text_features

            self.ensure_text_encoder_ready()

            print(f'Text list: {texts}')
            with torch.no_grad():
                text_features_list = []
                for text in texts:
                    text_tokens = self.tokenizer([text])
                    if isinstance(text_tokens, torch.Tensor):
                        text_tokens = text_tokens.long()
                    else:
                        text_tokens = torch.tensor(text_tokens, dtype=torch.long)
                    single_feature = self.text_encoder(text_tokens)
                    text_features_list.append(single_feature)
                text_features = torch.cat(text_features_list, dim=0)

            if not self.args.no_normalize:
                text_features = normalize_features(text_features)

            save_cached_text_features(self.args, text_features)
            self._set_status_plain('Preparing your text list…')
            return text_features
        finally:
            self.args.texts = previous_texts

    def apply_additional_texts(self):
        """Add new user-entered text lines at the top of the list (below the editor) and refresh the UI."""
        raw_lines = self.add_text_editor.toPlainText().splitlines()
        new_lines = [line.strip() for line in raw_lines if line.strip()]
        if not new_lines:
            self.show_error('Enter one or two text lines to add, then press Apply.')
            return

        existing = list(self.texts)
        appended = []
        for text in new_lines:
            if text not in existing and text not in appended:
                appended.append(text)

        if not appended:
            self._set_status_plain('Those text lines are already in the list.')
            self.add_text_editor.clear()
            return

        updated_texts = appended + existing

        self.apply_text_button.setEnabled(False)
        try:
            self.text_features = self.compute_text_features_for_texts(updated_texts)
            self.texts = updated_texts
            self.args.texts = list(updated_texts)
            self.rebuild_text_widgets()
            self.add_text_editor.clear()
            self._set_status_plain(
                f'Added {len(appended)} text quer{"y" if len(appended) == 1 else "ies"} to the list.'
            )
        except Exception as e:
            self.show_error(f'Adding text inputs failed: {e}')
        finally:
            self.apply_text_button.setEnabled(True)

    def setup_models(self):
        """Load ONNX text encoder, precompute text embeddings, and open async image encoder."""
        if not os.path.exists(self.args.text_encoder):
            self.show_error(f'Text encoder model file not found: {self.args.text_encoder}')
            return
        if not os.path.exists(self.args.image_encoder):
            self.show_error(f'Image encoder model file not found: {self.args.image_encoder}')
            return

        try:
            self.text_features = self.compute_text_features_for_texts(self.texts)
        except Exception as e:
            self.show_error(f'Text encoder tokenize/encode failed: {e}')
            return

        try:
            self.image_encoder = ImageEncoderAsync(self.args.image_encoder)
            self._set_status_plain('Starting live image encoder…')
        except Exception as e:
            self.show_error(f'Image encoder load failed: {e}')

    def setup_camera(self):
        """Start background capture; frames arrive via on_frame."""
        self.camera_thread = CameraThread(self.args.input, self.args.width, self.args.height, self.args.fps)
        self.camera_thread.frame_ready.connect(self.on_frame)
        self.camera_thread.start()
        self._set_status_plain('Camera on — matches will update live below.')

    def show_error(self, msg):
        self._set_status_plain('Setup issue — ' + str(msg))
        print(msg)

    def on_frame(self, frame):
        """Show preview; optionally enqueue async image encode (throttled by skip_frame)."""
        if self._shutting_down:
            return
        if frame is None:
            self.show_error(
                "We couldn't open the camera or video file. Check the device path or file location."
            )
            # Don't exit, just show error
            return

        preview_frame = frame if self._is_video_input else crop_center(
            frame, self.preview_width, self.preview_height
        )
        preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        lbl_w, lbl_h = self.video_label.width(), self.video_label.height()
        if lbl_w > 0 and lbl_h > 0:
            aspect_ratio = preview_rgb.shape[1] / preview_rgb.shape[0]
            if lbl_w / lbl_h > aspect_ratio:
                new_h = lbl_h
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = lbl_w
                new_h = int(new_w / aspect_ratio)
            preview_rgb = cv2.resize(preview_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        height, width = preview_rgb.shape[:2]
        bytes_per_line = preview_rgb.strides[0]
        qimg = QImage(preview_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        qimg = qimg.copy()
        self._preview_debug_count += 1
        should_log_preview = self._preview_debug_count <= 10 or self._preview_debug_count % 150 == 0
        if should_log_preview:
            if np.mean(preview_rgb) < 5.0:
                print('Warning: The camera is feeding mostly black frames. Please check your camera.')
            else:
                print(
                    'Preview debug: Extracted non-black frame. If the screen is black, '
                    'the issue is in PyQt rendering, not the camera feed. Otherwise, '
                    'the UI is displaying what the camera capture supplied.'
                )
        if qimg.isNull():
            self.show_error('Preview QImage creation failed for current frame.')
        else:
            pixmap = QPixmap.fromImage(qimg)
            if should_log_preview:
                print(f'Preview debug: pixmap_null={pixmap.isNull()}, size={pixmap.size()}')
            self.video_label.setPixmap(pixmap)

        frame_index = self._frame_index
        self._frame_index += 1

        if self.image_encoder is None or self.text_features is None:
            return

        infer_interval = max(0, self.args.skip_frame) + 1
        if frame_index % infer_interval != 0:
            return

        try:
            image_array = preprocess_frame(frame, self.transform)
            job_id = self._job_id
            self._job_id += 1
            req_id = self.image_encoder.encode_async(image_array, job_id)
            self._pending_jobs[job_id] = frame.copy()
            self._pending_reqs[job_id] = req_id
        except Exception as e:
            print(f'Preprocess/encode_async error: {e}')

    def on_timer(self):
        """Drain completed encodes: similarity per text, bar update, top-2 highlight if ≥ threshold."""
        if self._shutting_down:
            return
        if self.image_encoder is None or self.text_features is None:
            return

        while True:
            result = self.image_encoder.get_result(timeout=0)
            if result is None:
                break
            job_id, image_features = result
            req_id = self._pending_reqs.pop(job_id, None)
            self.image_encoder.notify_request_completed(req_id)
            frame_ref = self._pending_jobs.pop(job_id, None)
            if frame_ref is None:
                continue

            if not self.args.no_normalize:
                image_features = normalize_features(image_features.reshape(1, -1))[0]
            else:
                image_features = image_features.flatten()

            sim_scores = compute_similarity(self.text_features, image_features)
            for i, score in enumerate(sim_scores):
                self.text_widgets[i].update_score(score)

            # Rankings for highlight: only texts at or above HIGHLIGHT_MIN_SCORE qualify
            order = np.argsort(sim_scores)[::-1]
            eligible = [int(i) for i in order if float(sim_scores[i]) >= HIGHLIGHT_MIN_SCORE]
            for i, widget in enumerate(self.text_widgets):
                if len(eligible) > 0 and i == eligible[0]:
                    widget.set_highlight_rank(0)
                elif len(eligible) > 1 and i == eligible[1]:
                    widget.set_highlight_rank(1)
                else:
                    widget.set_highlight_rank(None)

            if len(eligible) > 0:
                bi = eligible[0]
                s1 = float(sim_scores[bi])
                t1 = html.escape(self.texts[bi], quote=True)
                part1 = (
                    f'<span style="color:{MATCH_COLOR_FIRST};font-weight:700">'
                    f'Best match:  {t1} - ({s1:.3f})</span>'
                )
                #if len(eligible) > 1:
                #    b2 = eligible[1]
                #    s2 = float(sim_scores[b2])
                #    t2 = html.escape(self.texts[b2], quote=True)
                #    part2 = (
                #        f'<span style="color:{MATCH_COLOR_SECOND};font-weight:600">'
                #        f'Second best:  {t2} - ({s2:.3f})</span>'
                #    )
                #    self._set_match_status_rich(part1 + '<br/>' + part2)
                #else:
                self._set_match_status_rich(part1)
            else:
                self._set_status_plain(
                    'No clear match yet - try a clearer view, better light, or move closer.'
                )

    def closeEvent(self, event):
        """Stop timers and threads, drain async results, then release DXNN (callback-safe)."""
        self._shutting_down = True

        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.stop()
            try:
                self.timer.timeout.disconnect(self.on_timer)
            except TypeError:
                pass

        if self.camera_thread is not None:
            try:
                self.camera_thread.frame_ready.disconnect(self.on_frame)
            except TypeError:
                pass
            self.camera_thread.stop()
            self.camera_thread = None

        if self.image_encoder is not None:
            # Drain GUI-thread queue so no pending slot runs after encoder teardown.
            while True:
                item = self.image_encoder.get_result(timeout=0)
                if item is None:
                    break
            self.image_encoder.close()
            self.image_encoder = None

        event.accept()


def extract_model_name_from_path(onnx_path: str) -> str:
    """Derive open_clip model name token from a *-something-text.onnx filename."""
    filename = os.path.basename(onnx_path)
    match = re.match(r'(.+?)-[^-]+-text\.onnx', filename)
    if match:
        return match.group(1)
    return 'ViT-L-14-quickgelu'


def main():
    """CLI entry: parse args, infer default model name, run Qt event loop."""
    parser = argparse.ArgumentParser(description='Realtime Camera Text Matcher Async GUI')
    parser.add_argument('--text-encoder', type=str, default='onnx/ViT-L-14-quickgelu-dfn2b-text.onnx')
    parser.add_argument('--image-encoder', type=str, default='dxnn/ViT-L-14-quickgelu-dfn2b.dxnn')
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--texts', type=str, nargs='+', required=True)
    parser.add_argument('--input', type=str, default=None, help='Camera device, camera index, or video file path')
    parser.add_argument('--camera', type=str, default='/dev/video0')
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--skip-frames', dest='skip_frame', type=int, default=2)

    parser.add_argument('--no-normalize', action='store_true')

    args = parser.parse_args()
    args.skip_frame = max(0, args.skip_frame)
    if args.input is None:
        args.input = args.camera

    if args.model_name is None:
        args.model_name = extract_model_name_from_path(args.text_encoder)

    app = QApplication(sys.argv)
    window = MainWindow(args)
    notify_launcher_ready()
    # Maximized fills the display while keeping the native title bar and close button
    # (Qt fullscreen would hide the frame).
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
