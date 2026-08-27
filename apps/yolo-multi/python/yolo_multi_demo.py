"""Multi-channel YOLO object detection demo (Python backend).

Mirrors the C++ backend (apps/yolo-multi/cpp) so both produce the same
detections from the same config: the same letterboxed BGR input tensor
(cpp/lib/utils/videostream.hpp), the same PPU bounding-box decode and
per-class NMS (cpp/src/demo_utils/yolo.cpp, nms.cpp), and the same board
layout and FPS accounting (cpp/src/yolo_multi_channels.cpp).
"""

import sys
import os
import argparse
import time
import math
import json
import threading
from collections import deque

import cv2
import numpy as np

from dx_engine import InferenceEngine

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

INPUT_CAPTURE_PERIOD_MS = 33
DISPLAY_WINDOW_NAME = "Object Detection"

_HAS_BATCHED_NMS = hasattr(cv2.dnn, "NMSBoxesBatched")

EXIT_BUTTON_WIDTH = 32
EXIT_BUTTON_HEIGHT = 28
EXIT_BUTTON_MARGIN = 8
TITLE_BAR_HEIGHT = 50

COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle",
    "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake",
    "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush",
]

# cpp/lib/utils/color_table.hpp : dxdemo::common::color_table
COLOR_TABLE = [
    (113, 129, 39), (133, 80, 164), (114, 122, 83), (172, 81, 99), (104, 56, 95), (86, 84, 37),
    (122, 89, 14), (65, 7, 80), (25, 102, 10), (109, 185, 90), (132, 110, 106), (85, 158, 169),
    (26, 185, 188), (17, 1, 103), (81, 144, 82), (184, 7, 92), (155, 81, 49), (69, 177, 179),
    (158, 187, 93), (73, 39, 13), (60, 50, 12), (33, 179, 16), (165, 69, 112), (63, 139, 15),
    (159, 191, 33), (32, 173, 182), (133, 113, 34), (34, 135, 90), (86, 34, 53), (190, 35, 141),
    (8, 171, 6), (112, 76, 118), (55, 60, 89), (88, 54, 15), (181, 75, 112), (38, 147, 42),
    (63, 52, 138), (149, 65, 128), (24, 103, 106), (45, 33, 168), (135, 136, 28), (108, 91, 86),
    (76, 11, 52), (189, 6, 142), (168, 81, 57), (148, 19, 55), (89, 101, 182), (179, 65, 44),
    (26, 33, 1), (26, 164, 122), (134, 63, 70), (82, 106, 137), (52, 118, 120), (42, 74, 129),
    (112, 147, 182), (50, 157, 22), (20, 50, 56), (177, 22, 2), (106, 100, 156), (42, 35, 21),
    (121, 8, 13), (28, 92, 142), (33, 118, 45), (30, 118, 105), (124, 185, 7), (146, 34, 46),
    (169, 184, 105), (5, 18, 22), (73, 71, 147), (91, 64, 181), (184, 39, 31), (33, 179, 164),
    (18, 50, 96), (106, 15, 95), (54, 68, 113), (112, 116, 136), (130, 139, 119), (34, 139, 31),
    (127, 6, 66), (2, 39, 62), (180, 99, 49), (155, 119, 49), (183, 50, 153), (3, 38, 125),
    (143, 87, 129), (40, 87, 49), (120, 62, 128), (148, 85, 73), (118, 144, 28), (24, 9, 29),
    (108, 45, 175), (64, 175, 81), (157, 19, 178), (190, 188, 74), (2, 114, 18), (96, 128, 62),
    (150, 3, 21), (95, 6, 0), (184, 20, 2), (185, 37, 122),
]


def make_yolo_layer_param(num_grid_x, num_grid_y, anchor_w, anchor_h, scale_x=0.0):
    """cpp/src/demo_utils/yolo_cfg.cpp : createYoloLayerParam()"""
    return {
        "num_grid_x": num_grid_x,
        "num_grid_y": num_grid_y,
        "anchor_w": anchor_w,
        "anchor_h": anchor_h,
        "scale_x": scale_x,
    }


# The three YOLOv5s/512 layer definitions shared by yolov5s_512 and yolov5s_512_ppu.
_YOLOV5S_512_LAYERS = [
    make_yolo_layer_param(64, 64, [10.0, 16.0, 33.0], [13.0, 30.0, 23.0]),
    make_yolo_layer_param(32, 32, [30.0, 62.0, 59.0], [61.0, 45.0, 119.0]),
    make_yolo_layer_param(16, 16, [116.0, 156.0, 373.0], [90.0, 198.0, 326.0]),
]

# cpp/src/demo_utils/yolo_cfg.cpp : the entries getYoloParameter() can return for
# the configs this demo ships. Unknown names fall back to yolov5s_512, exactly as
# the C++ does.
YOLO_PARAMS = {
    "yolov5s_512": {
        "width": 512, "height": 512,
        "conf_threshold": 0.25, "score_threshold": 0.3, "iou_threshold": 0.4,
        "num_classes": 80, "layers": _YOLOV5S_512_LAYERS, "class_names": COCO_CLASS_NAMES,
    },
    "yolov5s_512_ppu": {
        "width": 512, "height": 512,
        "conf_threshold": 0.25, "score_threshold": 0.3, "iou_threshold": 0.4,
        "num_classes": 80, "layers": _YOLOV5S_512_LAYERS, "class_names": COCO_CLASS_NAMES,
    },
    "yolov5s_320_ppu": {
        "width": 320, "height": 320,
        "conf_threshold": 0.25, "score_threshold": 0.3, "iou_threshold": 0.4,
        "num_classes": 80,
        "layers": [
            make_yolo_layer_param(40, 40, [10.0, 16.0, 33.0], [13.0, 30.0, 23.0]),
            make_yolo_layer_param(20, 20, [30.0, 62.0, 59.0], [61.0, 45.0, 119.0]),
            make_yolo_layer_param(10, 10, [116.0, 156.0, 373.0], [90.0, 198.0, 326.0]),
        ],
        "class_names": COCO_CLASS_NAMES,
    },
}


def get_yolo_parameter(model_name):
    """cpp/src/yolo_multi_channels.cpp : getYoloParameter()"""
    return YOLO_PARAMS.get(model_name, YOLO_PARAMS["yolov5s_512"])


def get_align_factor(length, based):
    """cpp/lib/utils/common_util.hpp : dxdemo::common::get_align_factor()"""
    return 0 if length % based == 0 else based - (length % based)


class Yolo:
    """PPU bounding-box decode + per-class NMS.

    Ports Yolo::ppu_post_processing() and Nms()/NmsOneClass() from
    cpp/src/demo_utils/. Boxes come back in NPU (letterboxed) coordinates as
    x1/y1/x2/y2, matching what the C++ hands to ObjectDetection::GetScalingBBox().
    """

    # sizeof(dxrt::DeviceBoundingBox_t)
    RECORD_BYTES = 32

    def __init__(self, param):
        self.cfg = param
        layers = param["layers"]
        self._stride_x = np.array(
            [param["width"] / l["num_grid_x"] for l in layers], dtype=np.float32)
        self._stride_y = np.array(
            [param["height"] / l["num_grid_y"] for l in layers], dtype=np.float32)
        self._anchor_w = np.array([l["anchor_w"] for l in layers], dtype=np.float32)
        self._anchor_h = np.array([l["anchor_h"] for l in layers], dtype=np.float32)
        self._scale_x = np.array([l["scale_x"] for l in layers], dtype=np.float32)
        self._num_layers = len(layers)
        self._num_boxes = self._anchor_w.shape[1]

    def post_proc(self, output):
        """Decode one DeviceBoundingBox_t array into (boxes, scores, labels)."""
        raw = output
        if raw.ndim == 3:  # (batch, num_elements, 32)
            raw = raw[0]
        if raw.size == 0:
            return self._empty()
        if raw.ndim != 2:
            raw = raw.reshape(-1, self.RECORD_BYTES)

        # DeviceBoundingBox_t (dxrt/dxrt_cxx_api.h), 32 bytes:
        #   float x, y, w, h; uint8 grid_y, grid_x, box_idx, layer_idx;
        #   float score; uint32 label; char padding[4];
        rec_f32 = raw.view(np.float32)
        rec_u32 = raw.view(np.uint32)

        keep = rec_f32[:, 5] > self.cfg["score_threshold"]
        if not keep.any():
            return self._empty()
        rec_f32 = rec_f32[keep]
        rec_u32 = rec_u32[keep]

        info = rec_u32[:, 4]
        grid_y = (info & 0xFF).astype(np.float32)
        grid_x = ((info >> 8) & 0xFF).astype(np.float32)
        box_idx = ((info >> 16) & 0xFF).astype(np.int64)
        layer_idx = ((info >> 24) & 0xFF).astype(np.int64)
        labels = rec_u32[:, 6].astype(np.int64)

        # The C++ warns and skips out-of-range labels; guard the grid indices too
        # so a corrupt record cannot index past the anchor table.
        valid = ((layer_idx < self._num_layers) & (box_idx < self._num_boxes)
                 & (labels < self.cfg["num_classes"]))
        if not valid.all():
            bad = int((labels >= self.cfg["num_classes"]).sum())
            if bad:
                print(f"[DXDEMO] [WARN] {bad} label id(s) out of range. "
                      "Please check the model output.", file=sys.stderr)
            if not valid.any():
                return self._empty()
            rec_f32 = rec_f32[valid]
            grid_y, grid_x = grid_y[valid], grid_x[valid]
            box_idx, layer_idx = box_idx[valid], layer_idx[valid]
            labels = labels[valid]

        stride_x = self._stride_x[layer_idx]
        stride_y = self._stride_y[layer_idx]
        scale_x_y = self._scale_x[layer_idx]

        x, y = rec_f32[:, 0], rec_f32[:, 1]
        w, h = rec_f32[:, 2], rec_f32[:, 3]
        scores = rec_f32[:, 5].copy()

        cx = np.where(scale_x_y == 0,
                      (x * 2.0 - 0.5 + grid_x),
                      (x * scale_x_y - 0.5 * (scale_x_y - 1) + grid_x)) * stride_x
        cy = np.where(scale_x_y == 0,
                      (y * 2.0 - 0.5 + grid_y),
                      (y * scale_x_y - 0.5 * (scale_x_y - 1) + grid_y)) * stride_y
        bw = np.square(w * 2.0) * self._anchor_w[layer_idx, box_idx]
        bh = np.square(h * 2.0) * self._anchor_h[layer_idx, box_idx]

        boxes = np.stack([cx - bw / 2.0, cy - bh / 2.0,
                          cx + bw / 2.0, cy + bh / 2.0], axis=1)
        keep_idx = self._nms(boxes, scores, labels)
        return boxes[keep_idx], scores[keep_idx], labels[keep_idx]

    def _nms(self, boxes, scores, labels):
        """cpp/src/demo_utils/nms.cpp : Nms() / NmsOneClass().

        Greedy per class, highest score first, dropping any lower-scored box
        whose IoU with a kept one exceeds the threshold. cv2.dnn.NMSBoxesBatched
        implements exactly that and runs outside the GIL, which matters because
        this executes on the engine's callback thread for every channel.
        """
        if len(boxes) == 0:
            return np.zeros(0, np.int64)
        if not _HAS_BATCHED_NMS:
            return self._nms_per_class(boxes, scores, labels)
        wh = np.empty((len(boxes), 4), np.float32)
        wh[:, 0] = boxes[:, 0]
        wh[:, 1] = boxes[:, 1]
        wh[:, 2] = boxes[:, 2] - boxes[:, 0]
        wh[:, 3] = boxes[:, 3] - boxes[:, 1]
        kept = cv2.dnn.NMSBoxesBatched(
            wh.tolist(), scores.tolist(), labels.tolist(),
            0.0, self.cfg["iou_threshold"])
        return np.asarray(kept, dtype=np.int64).ravel()

    def _nms_per_class(self, boxes, scores, labels):
        """Fallback for OpenCV builds without NMSBoxesBatched."""
        iou_threshold = self.cfg["iou_threshold"]
        kept = []
        for cls in np.unique(labels):
            idx = np.flatnonzero(labels == cls)
            idx = idx[np.argsort(-scores[idx], kind="stable")]
            cls_boxes = boxes[idx]
            area = ((cls_boxes[:, 2] - cls_boxes[:, 0])
                    * (cls_boxes[:, 3] - cls_boxes[:, 1]))
            alive = np.ones(len(idx), dtype=bool)
            for i in range(len(idx)):
                if not alive[i]:
                    continue
                kept.append(idx[i])
                rest = cls_boxes[i + 1:]
                if not rest.size:
                    break
                ovr_w = (np.minimum(cls_boxes[i, 2], rest[:, 2])
                         - np.maximum(cls_boxes[i, 0], rest[:, 0]))
                ovr_h = (np.minimum(cls_boxes[i, 3], rest[:, 3])
                         - np.maximum(cls_boxes[i, 1], rest[:, 1]))
                inter = np.where((ovr_w > 0) & (ovr_h > 0), ovr_w * ovr_h, 0.0)
                iou = inter / (area[i] + area[i + 1:] - inter)
                alive[i + 1:] &= ~(iou > iou_threshold)
        # Nms() sorts the merged result by descending score before returning.
        kept = np.array(kept, dtype=np.int64)
        return kept[np.argsort(-scores[kept], kind="stable")] if kept.size else kept

    @staticmethod
    def _empty():
        return (np.zeros((0, 4), np.float32),
                np.zeros(0, np.float32),
                np.zeros(0, np.int64))


class VideoStream:
    """Port of cpp/lib/utils/videostream.hpp.

    PRELOAD sources decode `num_frames` frames up front and loop over them;
    RUNTIME sources (camera / rtsp) read one frame per call.
    """

    PRELOAD = 0
    RUNTIME = 1
    REOPEN_INTERVAL_S = 3.0

    def __init__(self, input_type, src_path, num_frames, npu_size, dst_size, engine):
        # od.cpp maps the config string to AppInputType; anything unrecognised
        # (including "offline") is a plain video file.
        self._input_type = (input_type if input_type in ("camera", "image", "rtsp")
                            else "video")
        self._is_live = self._input_type in ("camera", "rtsp")
        self._src_path = src_path
        self._npu_w, self._npu_h = npu_size
        self._dst_w, self._dst_h = dst_size
        self._pre_load_num = num_frames
        self._last_reopen = 0.0

        self._video = self._open_capture()
        self._src_mode = (self.RUNTIME if self._is_live or num_frames <= 0
                          else self.PRELOAD)

        self.is_opened = self._video.isOpened()
        if not self.is_opened:
            print(f"Error: file {src_path} could not be opened.", file=sys.stderr)
            self._src_w, self._src_h = self._dst_w, self._dst_h
        else:
            self._src_w = int(self._video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._src_h = int(self._video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._src_w <= 0 or self._src_h <= 0:
            self._src_w, self._src_h = self._dst_w, self._dst_h

        # Preprocess geometry — identical arithmetic to the C++, including the
        # truncating cast on the resized size and the round(d +/- 0.1) pads.
        self.preproc_ratio = min(self._npu_w / self._src_w, self._npu_h / self._src_h)
        self._resize_w = int(self._src_w * self.preproc_ratio)
        self._resize_h = int(self._src_h * self.preproc_ratio)
        self._dw = (self._npu_w - self._resize_w) / 2.0
        self._dh = (self._npu_h - self._resize_h) / 2.0
        self._top = max(int(round(self._dh - 0.1)), 0)
        self._bottom = max(int(round(self._dh + 0.1)), 0)
        self._left = max(int(round(self._dw - 0.1)), 0)
        self._right = max(int(round(self._dw + 0.1)), 0)
        self._fill_pad_value = (-1 if (self._src_w, self._src_h) == (self._npu_w, self._npu_h)
                                else 114)

        # Input alignment: the NPU may want each row padded to a 64-byte boundary.
        npu_input_length = engine.get_input_size()
        self._need_preproc = npu_input_length != self._npu_w * self._npu_h * 3
        self._align_factor = (get_align_factor(self._npu_w * 3, 64)
                              if self._need_preproc else 0)
        self._npu_input_length = npu_input_length

        self._src_img = []
        self._pre_img = []
        self._src_img_runtime = None
        self._pre_img_runtime = None
        self._pre_get_cnt = 0
        self._out_get_cnt = 0

        if self._src_mode == self.PRELOAD:
            for _ in range(self._pre_load_num):
                if not self._preprocess():
                    break
            self._pre_load_num = len(self._pre_img)
            if self._pre_load_num == 0:
                self._src_mode = self.RUNTIME
            else:
                self._video.release()
                self._video = None

    @property
    def src_size(self):
        return self._src_w, self._src_h

    @property
    def padded_size(self):
        """ObjectDetection ctor: (npu - resized) / 2, or 0 when no letterbox."""
        if (self._src_w, self._src_h) == (self._npu_w, self._npu_h):
            return 0.0, 0.0
        return ((self._npu_w - self._resize_w) / 2.0,
                (self._npu_h - self._resize_h) / 2.0)

    def _open_capture(self):
        src = self._src_path
        if self._input_type == "camera" and str(src).isdigit():
            src = int(src)
        return cv2.VideoCapture(src)

    def _reopen_live_source(self):
        """A camera that is not ready at start-up, or one that is unplugged and
        plugged back in, would otherwise leave its tile black forever. Retry at
        most once every few seconds so a permanently dead device stays cheap."""
        now = time.monotonic()
        if now - self._last_reopen < self.REOPEN_INTERVAL_S:
            return
        self._last_reopen = now
        if self._video is not None:
            self._video.release()
        self._video = self._open_capture()
        self.is_opened = self._video.isOpened()

    def _img_capture(self):
        if self._video is None:
            return None
        ret, frame = self._video.read()
        if not ret or frame is None:
            if self._is_live:
                self._reopen_live_source()
                return None
            # Loop the file, as the C++ GStreamer pipeline does.
            self._video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._video.read()
            if not ret or frame is None:
                return None
        return frame

    def _img_pre_resize(self, src):
        if self._fill_pad_value < 0:
            return cv2.resize(src, (self._npu_w, self._npu_h),
                              interpolation=cv2.INTER_LINEAR)
        resized = cv2.resize(src, (self._resize_w, self._resize_h),
                             interpolation=cv2.INTER_LINEAR)
        return cv2.copyMakeBorder(
            resized, self._top, self._bottom, self._left, self._right,
            cv2.BORDER_CONSTANT,
            value=(self._fill_pad_value,) * 3)

    def _to_input_tensor(self, pre_img):
        """The NPU input is letterboxed BGR uint8 — the C++ passes
        AppInputFormat::IMAGE_BGR, so ImgCvtColor() is a no-op. Do not convert
        to RGB here or the model sees swapped channels."""
        if not self._need_preproc:
            return np.ascontiguousarray(pre_img)
        # data_pre_processing(): copy each row into a 64-byte aligned stride.
        buf = np.zeros(self._npu_input_length, dtype=np.uint8)
        copy_size = self._npu_w * 3
        stride = copy_size + self._align_factor
        flat = np.ascontiguousarray(pre_img).reshape(self._npu_h, copy_size)
        for y in range(self._npu_h):
            buf[y * stride:y * stride + copy_size] = flat[y]
        return buf

    def _preprocess(self):
        src_img = self._img_capture()
        if src_img is None:
            return False
        pre_img = self._to_input_tensor(self._img_pre_resize(src_img))
        if self._src_mode == self.RUNTIME:
            self._src_img_runtime = src_img
            self._pre_img_runtime = pre_img
        else:
            self._src_img.append(src_img)
            self._pre_img.append(pre_img)
        return True

    def get_input_stream(self):
        """Returns the input tensor, or None when the source yields no frame."""
        if self._src_mode == self.RUNTIME:
            if not self._preprocess():
                return None
            return self._pre_img_runtime
        img = self._pre_img[self._pre_get_cnt]
        self._pre_get_cnt = (self._pre_get_cnt + 1) % self._pre_load_num
        return img

    def get_output_stream(self, detections):
        """Resize the source frame to the tile size, then draw — the C++ draws
        boxes on the already-resized frame, so line widths and font sizes stay
        constant per tile."""
        if self._src_mode == self.RUNTIME:
            src = self._src_img_runtime
        else:
            src = self._src_img[self._out_get_cnt]
            self._out_get_cnt = (self._out_get_cnt + 1) % self._pre_load_num
        if src is None:
            return np.zeros((self._dst_h, self._dst_w, 3), np.uint8)

        if (self._src_w, self._src_h) == (self._dst_w, self._dst_h):
            dst = src.copy()
        else:
            dst = cv2.resize(src, (self._dst_w, self._dst_h),
                             interpolation=cv2.INTER_LINEAR)

        for xmin, ymin, xmax, ymax, label, name in detections:
            color = COLOR_TABLE[label % len(COLOR_TABLE)]
            cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), color, 2)
            (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(dst, (xmin, ymin - th), (xmin + tw, ymin), color, cv2.FILLED)
            cv2.putText(dst, name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255))
        return dst

    def release(self):
        if self._video is not None:
            self._video.release()
            self._video = None


class ObjectDetection:
    """Port of cpp/src/od.cpp — one capture/inference thread per channel, all
    sharing a single InferenceEngine and posting results through its callback."""

    def __init__(self, engine, yolo_param, video_src, channel,
                 dst_width, dst_height, pos_x, pos_y, num_frames):
        self._ie = engine
        self._channel = channel + 1
        self._dst_width = dst_width
        self._dst_height = dst_height
        self._pos_x = pos_x
        self._pos_y = pos_y
        self._video_src = video_src

        self._lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._toggle_drawing = True
        self._detections = []
        self._processed_count = 0
        self._ret_processed_count = 0

        self._result_frame = np.zeros((dst_height, dst_width, 3), np.uint8)
        self.is_blank = video_src is None

        if self.is_blank:
            self.yolo = None
            self._vstream = None
            self._load_logo()
            return

        src_path, src_type = video_src
        input_shape = engine.get_input_tensors_info()[0]["shape"]
        npu_size = (int(input_shape[1]), int(input_shape[1]))
        self._vstream = VideoStream(src_type, src_path, num_frames,
                                    npu_size, (dst_width, dst_height), engine)
        self.yolo = Yolo(yolo_param)
        self._class_names = yolo_param["class_names"]

        src_w, src_h = self._vstream.src_size
        self._postproc_padded = self._vstream.padded_size
        preproc_ratio = self._vstream.preproc_ratio
        self._postproc_scale = ((dst_width / src_w) / preproc_ratio,
                                (dst_height / src_h) / preproc_ratio)
        self._cam_ui = self._build_camera_ui() if src_type == "camera" else None

    def _load_logo(self):
        """Blank tiles show the DX logo (od.cpp fill-blank constructor)."""
        for candidate in ("../../../workspace/assets/yolo-multi/sample/dx_colored_log.png",
                          "../../workspace/assets/yolo-multi/sample/dx_colored_log.png"):
            if os.path.exists(candidate):
                logo = cv2.imread(candidate, cv2.IMREAD_COLOR)
                if logo is not None:
                    self._result_frame = cv2.resize(
                        logo, (self._dst_width, self._dst_height),
                        interpolation=cv2.INTER_LINEAR)
                return

    def _build_camera_ui(self):
        """od.cpp threadFunc(): yellow border plus a 'LIVE' pill on camera tiles."""
        cols, rows = self._dst_width, self._dst_height
        font_scale = min(0.62, max(0.38, cols * 0.0011))
        font_thick = 1
        (tw, th), _ = cv2.getTextSize("Live", cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, font_thick)
        pad_x = max(5, int(7 * font_scale / 0.55))
        pad_y = max(3, int(5 * font_scale / 0.55))
        margin = max(4, cols // 80)
        dot_r = max(2, int(4 * font_scale / 0.55))
        inner_gap = max(3, int(5 * font_scale / 0.55))
        pill_w = pad_x + dot_r * 2 + inner_gap + tw + pad_x
        pill_h = pad_y + th + pad_y
        x2 = cols - margin
        x1 = x2 - pill_w
        y1 = margin
        y2 = y1 + pill_h
        if not (x1 >= 0 and y2 <= rows and pill_w + 2 * margin <= cols):
            return {"show_pill": False}
        return {
            "show_pill": True,
            "pill": (x1, y1, pill_w, pill_h),
            "dot_center": (x1 + pad_x + dot_r, y1 + pill_h // 2),
            "dot_r": dot_r,
            "font_scale": font_scale,
            "font_thick": font_thick,
            "text_org": (x1 + pad_x + dot_r * 2 + inner_gap, y1 + pad_y + th),
        }

    def _scale_detections(self, boxes, labels):
        """od.cpp GetScalingBBox(): NPU coords -> tile coords."""
        pad_w, pad_h = self._postproc_padded
        sx, sy = self._postproc_scale
        names = self._class_names
        return [(int((x1 - pad_w) * sx), int((y1 - pad_h) * sy),
                 int((x2 - pad_w) * sx), int((y2 - pad_h) * sy),
                 label, names[label])
                for (x1, y1, x2, y2), label in zip(boxes.tolist(), labels.tolist())]

    def post_proc(self, output):
        """Called from the engine's completion callback (od.cpp PostProc())."""
        boxes, _scores, labels = self.yolo.post_proc(output)
        detections = self._scale_detections(boxes, labels)
        with self._lock:
            self._detections = detections
            self._processed_count += 1
            self._ret_processed_count += 1

    def _thread_func(self, period):
        period_s = period / 1000.0
        while not self._stop.is_set():
            proc_start = time.monotonic()
            input_tensor = self._vstream.get_input_stream()
            if input_tensor is None:
                # A dead source (unplugged camera, end of stream) yields no
                # frame. Idle this channel and retry rather than killing the
                # thread, which would freeze the tile for good.
                self._stop.wait(period_s)
                continue

            self._ie.run_async([input_tensor], user_arg=self)

            with self._lock:
                detections = self._detections if self._toggle_drawing else []
            frame = self._vstream.get_output_stream(detections)

            if self._cam_ui is not None:
                cv2.rectangle(frame, (0, 0),
                              (self._dst_width - 1, self._dst_height - 1),
                              (0, 255, 255), 3)
                if self._cam_ui["show_pill"]:
                    cv2.rectangle(frame, self._cam_ui["pill"], (248, 248, 255),
                                  cv2.FILLED)
                    cv2.circle(frame, self._cam_ui["dot_center"], self._cam_ui["dot_r"],
                               (0, 0, 255), cv2.FILLED, cv2.LINE_AA)
                    cv2.putText(frame, "LIVE", self._cam_ui["text_org"],
                                cv2.FONT_HERSHEY_SIMPLEX, self._cam_ui["font_scale"],
                                (28, 28, 32), self._cam_ui["font_thick"], cv2.LINE_AA)

            cv2.rectangle(frame, (0, 0), (40, 20), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, str(self._channel), (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

            with self._lock:
                published = self._processed_count > 0
            if published:
                with self._frame_lock:
                    self._result_frame = frame

            sleep_for = period_s - (time.monotonic() - proc_start)
            if 0 < sleep_for <= period_s:
                self._stop.wait(sleep_for)

    def run(self, period):
        if self.is_blank:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_func, args=(period,),
                                        daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._vstream is not None:
            self._vstream.release()

    def toggle(self):
        with self._lock:
            self._toggle_drawing = not self._toggle_drawing

    def result_frame(self):
        with self._frame_lock:
            return self._result_frame

    def position(self):
        return self._pos_x, self._pos_y

    def resolution(self):
        return self._dst_width, self._dst_height

    def get_post_process_count(self):
        with self._lock:
            return self._ret_processed_count


class AppConfig:
    """Port of ApplicationJsonParser() in cpp/src/yolo_multi_channels.cpp."""

    def __init__(self, config_path):
        with open(config_path, "r") as f:
            doc = json.load(f)

        self.application_type = 1 if doc.get("usage") == "multi" else 0
        self.model_path = doc["model_path"]
        self.model_name = doc["model_name"]

        display = doc["display_config"]
        self.display_label = display["display_label"]
        self.full_screen = "output_width" not in display
        if self.full_screen:
            self.board_width, self.board_height = self._framebuffer_size()
        else:
            self.board_width = int(display["output_width"])
            self.board_height = int(display["output_height"])
        self.input_capture_period_ms = int(
            display.get("capture_period", INPUT_CAPTURE_PERIOD_MS))
        self.is_show_fps = bool(display.get("show_fps", True))
        self.is_fill_blank = bool(display.get("fill_blank", True))
        self.is_expand_mode = bool(display.get("expand_mode", False))

        self.video_sources = []
        self.pre_saved_frame_count = []
        for entry in doc["video_sources"]:
            self.video_sources.append((entry[0], entry[1]))
            if entry[1] == "offline":
                self.pre_saved_frame_count.append(int(entry[2]) if len(entry) == 3 else 0)
            else:
                self.pre_saved_frame_count.append(-1)

    @staticmethod
    def _framebuffer_size():
        try:
            with open("/sys/class/graphics/fb0/virtual_size") as f:
                w, h = f.read().strip().split(",")
                return int(w), int(h)
        except (OSError, ValueError):
            print("Failed to open framebuffer info, It will be set FHD size")
            return 1920, 1080


def expand_position_index(i, num_sources, div):
    """Port of the expand-mode tile remap in yolo_multi_channels.cpp."""
    layouts = {
        33: ([(14, 0), (18, 2), (32, 4)], 14),
        41: ([(16, 0), (20, 3), (24, 6), (40, 9)], 16),
        61: ([(27, 0), (33, 2), (60, 4)], 27),
        73: ([(30, 0), (36, 3), (42, 6), (72, 9)], 30),
    }
    bands, last = layouts[num_sources]
    for limit, offset in bands:
        if i < limit:
            return i + offset
    return last


class DemoWindow(QMainWindow):
    """Composites every channel into one board, like the C++ cv::imshow loop."""

    FPS_WARMUP_MS = 1000

    def __init__(self, config_path, show_exit_button=False, window_size=60.0):
        super().__init__()
        self.config = AppConfig(config_path)
        self.window_size = window_size
        self.show_exit_button = show_exit_button
        self._closed = False

        self.setWindowTitle(self.config.display_label or DISPLAY_WINDOW_NAME)
        board_w = self.config.board_width
        board_h = self.config.board_height
        self.resize(board_w, board_h)

        self.exit_button_rect = (
            board_w - EXIT_BUTTON_WIDTH - EXIT_BUTTON_MARGIN,
            (TITLE_BAR_HEIGHT - EXIT_BUTTON_HEIGHT) // 2,
            EXIT_BUTTON_WIDTH, EXIT_BUTTON_HEIGHT)

        num_sources = len(self.config.video_sources)
        div = int(math.ceil(math.sqrt(num_sources)))
        div_w = board_w // div
        div_h = board_h // div
        if self.config.is_expand_mode and num_sources not in (33, 41, 61, 73):
            self.config.is_expand_mode = False

        self.engine = InferenceEngine(self.config.model_path)
        if not self.engine.is_ppu():
            raise SystemExit(
                "[DXDEMO] [ER] The Python backend implements the PPU (BBOX) output "
                f"path only; '{self.config.model_path}' is not a PPU model. "
                "Use the C++ backend for raw-tensor models.")
        self.yolo_param = get_yolo_parameter(self.config.model_name)

        self.apps = []
        if self.config.is_expand_mode:
            scale = 3 if num_sources in (41, 73) else 2
            for i in range(num_sources):
                pos = expand_position_index(i, num_sources, div)
                last = i == num_sources - 1
                self.apps.append(ObjectDetection(
                    self.engine, self.yolo_param, self.config.video_sources[i], i,
                    div_w * (scale if last else 1), div_h * (scale if last else 1),
                    div_w * (pos % div), div_h * (pos // div),
                    self.config.pre_saved_frame_count[i]))
        else:
            for i in range(num_sources):
                self.apps.append(ObjectDetection(
                    self.engine, self.yolo_param, self.config.video_sources[i], i,
                    div_w, div_h, div_w * (i % div), div_h * (i // div),
                    self.config.pre_saved_frame_count[i]))
            if self.config.is_fill_blank:
                for i in range(num_sources, div * div):
                    self.apps.append(ObjectDetection(
                        self.engine, self.yolo_param, None, i,
                        div_w, div_h, div_w * (i % div), div_h * (i // div), 0))

        self.engine.register_callback(self._on_inference_done)

        self.out_frame = np.zeros((board_h, board_w, 3), np.uint8)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.setCentralWidget(self.label)

        for app in self.apps:
            app.run(self.config.input_capture_period_ms)

        self.num_sources = num_sources
        self.start_time = time.monotonic()
        self.calc_fps = False
        self.last_counts = []
        self.timestamped_counts = deque()
        self.display_fps = 0.0
        self.display_fps_per_ch = 0.0
        self.have_fps_display = False
        self.last_fps_display_update = time.monotonic()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(self.config.input_capture_period_ms)

    @staticmethod
    def _on_inference_done(outputs, user_arg):
        """Engine completion callback — mirrors ie->RegisterCallback() in the C++."""
        try:
            user_arg.post_proc(outputs[0])
        except Exception as exc:  # a raising callback would abort the engine thread
            print(f"[DXDEMO] [ER] post-processing failed: {exc}", file=sys.stderr)
        return 0

    def _update_fps(self):
        """Sliding-window FPS over per-channel post-process counts, as the C++
        main loop computes it."""
        now = time.monotonic()
        if not self.calc_fps:
            if (now - self.start_time) * 1000.0 > self.FPS_WARMUP_MS:
                self.calc_fps = True
                self.last_counts = [a.get_post_process_count()
                                    for a in self.apps[:self.num_sources]]
            return

        check_sum = 0
        for i, app in enumerate(self.apps[:self.num_sources]):
            current = app.get_post_process_count()
            check_sum += max(0, current - self.last_counts[i])
            self.last_counts[i] = current
        self.timestamped_counts.append((now, check_sum))

        if self.window_size > 0:
            cutoff = now - self.window_size
            while self.timestamped_counts and self.timestamped_counts[0][0] < cutoff:
                self.timestamped_counts.popleft()

        frame_count = sum(c for _, c in self.timestamped_counts)
        if len(self.timestamped_counts) > 1:
            span = self.timestamped_counts[-1][0] - self.timestamped_counts[0][0]
            result_fps = frame_count / span if span > 0 else frame_count
        else:
            result_fps = float(frame_count)

        if not self.have_fps_display or (now - self.last_fps_display_update) >= 0.1:
            self.display_fps = result_fps
            self.display_fps_per_ch = result_fps / max(1, self.num_sources)
            self.last_fps_display_update = now
            self.have_fps_display = True

    def _draw_exit_button(self):
        x, y, w, h = self.exit_button_rect
        cv2.rectangle(self.out_frame, (x, y), (x + w, y + h), (60, 60, 60), cv2.FILLED)
        cv2.rectangle(self.out_frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
        pad = 8
        cv2.line(self.out_frame, (x + pad, y + pad), (x + w - pad, y + h - pad),
                 (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(self.out_frame, (x + w - pad, y + pad), (x + pad, y + h - pad),
                 (255, 255, 255), 2, cv2.LINE_AA)

    def update_frames(self):
        for app in self.apps:
            x, y = app.position()
            w, h = app.resolution()
            frame = app.result_frame()
            fh, fw = frame.shape[:2]
            h = min(h, self.out_frame.shape[0] - y, fh)
            w = min(w, self.out_frame.shape[1] - x, fw)
            if h > 0 and w > 0:
                self.out_frame[y:y + h, x:x + w] = frame[:h, :w]

        self._update_fps()

        if self.config.is_show_fps:
            board_w = self.out_frame.shape[1]
            cv2.rectangle(self.out_frame, (0, 0), (500, 50), (0, 0, 255), cv2.FILLED)
            cv2.putText(self.out_frame, f" {len(self.apps)}ch Real-time Processing ",
                        (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.rectangle(self.out_frame, (500, 0), (board_w, 50), (0, 0, 0), cv2.FILLED)
            if self.calc_fps and self.have_fps_display:
                sub = (f"        AI Model : {self.config.model_name}         "
                       f"FPS : {self.display_fps:.2f}      "
                       f"FPS/Stream: {self.display_fps_per_ch:.2f}   ")
            else:
                sub = (f"        AI Model : {self.config.model_name}         "
                       "FPS : --      FPS/Stream: --   ")
            cv2.putText(self.out_frame, sub, (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2, cv2.LINE_AA)

        if self.show_exit_button:
            self._draw_exit_button()

        h, w, ch = self.out_frame.shape
        qimg = QImage(self.out_frame.data, w, h, ch * w, QImage.Format.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(qimg.copy()))

    def _board_point(self, pos):
        """Map a widget click back onto the board image, which is letterboxed
        into the label by AlignCenter."""
        pix = self.label.pixmap()
        if pix is None or pix.isNull():
            return None
        lbl = self.label
        off_x = lbl.x() + (lbl.width() - pix.width()) // 2
        off_y = lbl.y() + (lbl.height() - pix.height()) // 2
        return pos.x() - off_x, pos.y() - off_y

    def mousePressEvent(self, event):
        if not self.show_exit_button:
            return
        point = self._board_point(event.position().toPoint())
        if point is None:
            return
        x, y, w, h = self.exit_button_rect
        if x <= point[0] <= x + w and y <= point[1] <= y + h:
            self.close()

    def closeEvent(self, event):
        if self._closed:
            event.accept()
            return
        self._closed = True
        self.timer.stop()
        # Stop the channel threads first so no new jobs are queued, then let the
        # in-flight ones drain before dropping the callback: it still holds
        # references to the ObjectDetection objects it posts results to.
        for app in self.apps:
            app.stop()
        time.sleep(1.0)
        self.engine.register_callback(None)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Q, Qt.Key.Key_Escape):
            self.close()
        elif event.key() == Qt.Key.Key_T:
            for app in self.apps:
                app.toggle()


def notify_launcher_ready():
    path = os.environ.get("DX_LAUNCHER_READY_FILE")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write("ready\n")
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="yolo multi channels application")
    parser.add_argument("-c", "--config", type=str, required=True,
                        help="use config json file for run application")
    parser.add_argument("--exit-btn", action="store_true",
                        help="show a small exit button in the top-right title bar")
    parser.add_argument("--window_size", type=float, default=60.0,
                        help="FPS by average over the last {window_size} seconds")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = DemoWindow(args.config, show_exit_button=args.exit_btn,
                     window_size=args.window_size)
    win.showFullScreen()
    notify_launcher_ready()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
