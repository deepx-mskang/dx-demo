import argparse
import ctypes
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QPoint, QRect, QSize, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QLinearGradient, QMouseEvent, QKeyEvent, QPainterPath
from PyQt5.QtWidgets import QApplication, QWidget

import dx_engine

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_PALM_MODEL = "../../../workspace/models/hands/hand-detector_192x192.dxnn"
DEFAULT_LANDMARK_MODEL = "../../../workspace/models/hands/HandLandmarkLite.dxnn"
DEFAULT_POSE_MODEL = "../../../workspace/models/hands/yolo26s-pose.dxnn"

PALM_WIDTH, PALM_HEIGHT, PALM_CHANNELS = 192, 192, 3
PALM_ANCHOR_COUNT, PALM_BOX_COORDS, PALM_KEYPOINT_COUNT = 2016, 18, 7
LANDMARK_WIDTH, LANDMARK_HEIGHT, LANDMARK_CHANNELS = 224, 224, 3
HAND_LANDMARK_COUNT = 21

QUEUE_SIZE = 3

# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class NormalizedRect:
    x_center: float = 0.0
    y_center: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0

@dataclass
class PalmDetection:
    box: List[float] # [x, y, w, h]
    keypoints: List[Tuple[float, float]]
    hand_roi: NormalizedRect = None
    score: float = 0.0

@dataclass
class HandLandmarkResult:
    palm: PalmDetection
    landmarks: List[Tuple[float, float, float]] # (x, y, z)
    world_landmarks: List[Tuple[float, float, float]]
    handedness: str = "Unknown"
    handedness_score: float = 0.0
    confidence: float = 0.0

@dataclass
class PoseKeypoint:
    x: float
    y: float
    confidence: float

@dataclass
class PoseResult:
    box: List[float] # [x1, y1, x2, y2]
    confidence: float
    keypoints: List[PoseKeypoint]

@dataclass
class FramePacket:
    id: int
    frame: np.ndarray
    captured_at: float

@dataclass
class PosePacket:
    frame_id: int
    poses: List[PoseResult]
    inference_ms: float

@dataclass
class HandPacket:
    frame_id: int
    palms: List[PalmDetection]
    hands: List[HandLandmarkResult]

@dataclass
class ResultBundle:
    has_pose: bool = False
    pose: PosePacket = None
    has_hand: bool = False
    hand: HandPacket = None

@dataclass
class PreprocessInfo:
    scale_x: float = 1.0
    scale_y: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0

@dataclass
class PosePreprocessInfo:
    pad_x: int = 0
    pad_y: int = 0
    scale: float = 1.0
    original_width: int = 0
    original_height: int = 0
    input_width: int = 0
    input_height: int = 0

# ==============================================================================
# Queue & Sync Utilities
# ==============================================================================
class BoundedQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = deque()
        self.cond = threading.Condition()
        self.closed = False

    def push(self, item):
        with self.cond:
            if self.closed:
                return False
            while len(self.items) >= self.max_size:
                self.items.popleft()
            self.items.append(item)
            self.cond.notify()
        return True

    def pop_for(self, timeout):
        with self.cond:
            if not self.items and not self.closed:
                self.cond.wait(timeout)
            if not self.items:
                return None
            return self.items.popleft()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def is_closed(self):
        with self.cond:
            return self.closed

class ResultStore:
    def __init__(self):
        self.cond = threading.Condition()
        self.pose_results = {}
        self.hand_results = {}
        self.latest_pose = None
        self.latest_hand = None

    def set_pose(self, packet: PosePacket):
        with self.cond:
            self.latest_pose = packet
            self.pose_results[packet.frame_id] = packet
            self.cond.notify_all()

    def set_hand(self, packet: HandPacket):
        with self.cond:
            self.latest_hand = packet
            self.hand_results[packet.frame_id] = packet
            self.cond.notify_all()

    def wait_for(self, frame_id, need_pose, need_hand, timeout):
        start_time = time.time()
        with self.cond:
            while True:
                ready = True
                if need_pose and frame_id not in self.pose_results:
                    ready = False
                if need_hand and frame_id not in self.hand_results:
                    ready = False
                
                if ready or (time.time() - start_time) > timeout:
                    break
                self.cond.wait(timeout)

            bundle = ResultBundle()
            
            if frame_id in self.pose_results:
                bundle.pose = self.pose_results[frame_id]
                bundle.has_pose = True
            elif self.latest_pose and self.latest_pose.frame_id <= frame_id:
                bundle.pose = self.latest_pose
                bundle.has_pose = True
                
            if frame_id in self.hand_results:
                bundle.hand = self.hand_results[frame_id]
                bundle.has_hand = True
            elif self.latest_hand and self.latest_hand.frame_id <= frame_id:
                bundle.hand = self.latest_hand
                bundle.has_hand = True
                
            return bundle

    def prune_before(self, min_frame_id):
        with self.cond:
            keys = list(self.pose_results.keys())
            for k in keys:
                if k < min_frame_id:
                    del self.pose_results[k]
            keys = list(self.hand_results.keys())
            for k in keys:
                if k < min_frame_id:
                    del self.hand_results[k]

class FpsCounter:
    def __init__(self):
        self.frames = 0
        self.fps = 0.0
        self.last = time.time()

    def update(self):
        self.frames += 1
        now = time.time()
        elapsed = now - self.last
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.last = now
        return self.fps

# ==============================================================================
# Math & Geometry
# ==============================================================================
def sigmoid(value):
    value = np.clip(value, -100.0, 100.0)
    return 1.0 / (1.0 + np.exp(-value))

def normalize_radians(angle):
    return angle - 2.0 * math.pi * math.floor((angle + math.pi) / (2.0 * math.pi))

def intersection_area(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def intersection_over_union(a, b):
    inter = intersection_area(a, b)
    area_a = max(0.0, a[2]) * max(0.0, a[3])
    area_b = max(0.0, b[2]) * max(0.0, b[3])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0

# ==============================================================================
# Palm Anchors & NMS
# ==============================================================================
def generate_palm_anchors():
    kNumLayers = 4
    kMinScale = 0.1484375
    kMaxScale = 0.75
    kAnchorOffsetX = 0.5
    kAnchorOffsetY = 0.5
    kInterpolatedScaleAspectRatio = 1.0
    strides = [8, 16, 16, 16]

    def calculate_scale(stride_index):
        if kNumLayers == 1: return (kMinScale + kMaxScale) * 0.5
        return kMinScale + (kMaxScale - kMinScale) * float(stride_index) / float(kNumLayers - 1)

    anchors = []
    layer_id = 0
    while layer_id < kNumLayers:
        anchor_widths = []
        anchor_heights = []
        aspect_ratios = []
        scales = []
        last_same_stride_layer = layer_id

        while last_same_stride_layer < kNumLayers and strides[last_same_stride_layer] == strides[layer_id]:
            scale = calculate_scale(last_same_stride_layer)
            aspect_ratios.append(1.0)
            scales.append(scale)
            next_scale = 1.0 if last_same_stride_layer == kNumLayers - 1 else calculate_scale(last_same_stride_layer + 1)
            aspect_ratios.append(kInterpolatedScaleAspectRatio)
            scales.append(math.sqrt(scale * next_scale))
            last_same_stride_layer += 1

        for i in range(len(aspect_ratios)):
            ratio_sqrt = math.sqrt(aspect_ratios[i])
            anchor_heights.append(scales[i] / ratio_sqrt)
            anchor_widths.append(scales[i] * ratio_sqrt)

        feature_map_height = math.ceil(PALM_HEIGHT / strides[layer_id])
        feature_map_width = math.ceil(PALM_WIDTH / strides[layer_id])

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_widths)):
                    anchors.append((
                        (float(x) + kAnchorOffsetX) / float(feature_map_width),
                        (float(y) + kAnchorOffsetY) / float(feature_map_height),
                        1.0,
                        1.0
                    ))
        layer_id = last_same_stride_layer
    return anchors

def weighted_candidate(detections, cluster_indices):
    top = detections[cluster_indices[0]]
    out = PalmDetection(box=top.box.copy(), keypoints=[list(kp) for kp in top.keypoints], score=top.score)
    total_score = 0.0
    xmin, ymin, xmax, ymax = 0.0, 0.0, 0.0, 0.0
    keypoints = [[0.0, 0.0] for _ in range(PALM_KEYPOINT_COUNT)]

    for idx in cluster_indices:
        candidate = detections[idx]
        weight = candidate.score
        total_score += weight
        xmin += candidate.box[0] * weight
        ymin += candidate.box[1] * weight
        xmax += (candidate.box[0] + candidate.box[2]) * weight
        ymax += (candidate.box[1] + candidate.box[3]) * weight
        for k in range(PALM_KEYPOINT_COUNT):
            keypoints[k][0] += candidate.keypoints[k][0] * weight
            keypoints[k][1] += candidate.keypoints[k][1] * weight

    if total_score <= 0.0:
        return out
    xmin /= total_score
    ymin /= total_score
    xmax /= total_score
    ymax /= total_score
    out.box = [xmin, ymin, max(0.0, xmax - xmin), max(0.0, ymax - ymin)]
    for k in range(PALM_KEYPOINT_COUNT):
        out.keypoints[k][0] = keypoints[k][0] / total_score
        out.keypoints[k][1] = keypoints[k][1] / total_score
    return out

def weighted_non_max_suppression(detections, threshold, max_detections):
    detections.sort(key=lambda x: x.score, reverse=True)
    remaining = list(range(len(detections)))
    selected = []

    while remaining and len(selected) < max_detections:
        top_idx = remaining[0]
        cluster = []
        next_remaining = []
        for idx in remaining:
            if intersection_over_union(detections[idx].box, detections[top_idx].box) > threshold:
                cluster.append(idx)
            else:
                next_remaining.append(idx)
        selected.append(weighted_candidate(detections, cluster))
        remaining = next_remaining
    return selected

def preprocess_palm_frame(bgr, keep_aspect, info):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    input_img = np.zeros((PALM_HEIGHT, PALM_WIDTH, 3), dtype=np.uint8)

    if keep_aspect:
        scale = min(PALM_WIDTH / rgb.shape[1], PALM_HEIGHT / rgb.shape[0])
        resized_w = max(1, int(round(rgb.shape[1] * scale)))
        resized_h = max(1, int(round(rgb.shape[0] * scale)))
        pad_x = (PALM_WIDTH - resized_w) // 2
        pad_y = (PALM_HEIGHT - resized_h) // 2
        resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        input_img[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w] = resized
        info.scale_x = scale
        info.scale_y = scale
        info.pad_x = float(pad_x)
        info.pad_y = float(pad_y)
    else:
        input_img = cv2.resize(rgb, (PALM_WIDTH, PALM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        info.scale_x = PALM_WIDTH / rgb.shape[1]
        info.scale_y = PALM_HEIGHT / rgb.shape[0]
        info.pad_x = 0.0
        info.pad_y = 0.0

    return np.ascontiguousarray(input_img, dtype=np.uint8)

def make_hand_roi(box, keypoints, frame_size):
    image_w, image_h = float(frame_size[0]), float(frame_size[1])
    roi = NormalizedRect()
    roi.x_center = (box[0] + box[2] * 0.5) / image_w
    roi.y_center = (box[1] + box[3] * 0.5) / image_h
    roi.width = box[2] / image_w
    roi.height = box[3] / image_h

    start = keypoints[0] # kWristCenterKeypoint
    end = keypoints[2]   # kMiddleFingerMcpKeypoint
    
    # kHandRoiTargetAngle = pi * 0.5
    roi.rotation = normalize_radians((math.pi * 0.5) - math.atan2(-(end[1] - start[1]), end[0] - start[0]))

    width = roi.width
    height = roi.height
    
    # kHandRoiShiftX = 0.0, kHandRoiShiftY = -0.5
    x_shift = (image_w * width * 0.0 * math.cos(roi.rotation) - image_h * height * -0.5 * math.sin(roi.rotation)) / image_w
    y_shift = (image_w * width * 0.0 * math.sin(roi.rotation) + image_h * height * -0.5 * math.cos(roi.rotation)) / image_h
    roi.x_center += x_shift
    roi.y_center += y_shift

    long_side = max(width * image_w, height * image_h)
    width = long_side / image_w
    height = long_side / image_h
    
    # kHandRoiScaleX = 2.6, kHandRoiScaleY = 2.6
    roi.width = width * 2.6
    roi.height = height * 2.6
    return roi

def hand_roi_corners(roi, frame_size):
    cx = roi.x_center * frame_size[0]
    cy = roi.y_center * frame_size[1]
    w = roi.width * frame_size[0]
    h = roi.height * frame_size[1]
    cos_r = math.cos(roi.rotation)
    sin_r = math.sin(roi.rotation)
    
    local = [
        (-w * 0.5, -h * 0.5),
        (w * 0.5, -h * 0.5),
        (w * 0.5, h * 0.5),
        (-w * 0.5, h * 0.5)
    ]
    
    corners = []
    for lx, ly in local:
        nx = cx + lx * cos_r - ly * sin_r
        ny = cy + lx * sin_r + ly * cos_r
        corners.append((nx, ny))
    return corners

def decode_palm_detections(box_tensor, score_tensor, anchors, prep, frame_size, confidence, nms_threshold, max_hands):
    candidates = []
    
    def model_to_frame_x(normalized_x):
        return (normalized_x * float(PALM_WIDTH) - prep.pad_x) / prep.scale_x
        
    def model_to_frame_y(normalized_y):
        return (normalized_y * float(PALM_HEIGHT) - prep.pad_y) / prep.scale_y

    for i in range(PALM_ANCHOR_COUNT):
        score = sigmoid(score_tensor[0, i, 0])
        if score < confidence:
            continue

        box = box_tensor[0, i]
        anchor = anchors[i]
        x_center = box[0] / float(PALM_WIDTH) * anchor[2] + anchor[0]
        y_center = box[1] / float(PALM_HEIGHT) * anchor[3] + anchor[1]
        box_w = box[2] / float(PALM_WIDTH) * anchor[2]
        box_h = box[3] / float(PALM_HEIGHT) * anchor[3]
        
        if box_w <= 0.0 or box_h <= 0.0:
            continue

        candidate = PalmDetection(
            box=[x_center - box_w * 0.5, y_center - box_h * 0.5, box_w, box_h],
            keypoints=[[0.0, 0.0] for _ in range(PALM_KEYPOINT_COUNT)],
            score=score
        )
        
        for k in range(PALM_KEYPOINT_COUNT):
            offset = 4 + k * 2
            candidate.keypoints[k][0] = box[offset] / float(PALM_WIDTH) * anchor[2] + anchor[0]
            candidate.keypoints[k][1] = box[offset + 1] / float(PALM_HEIGHT) * anchor[3] + anchor[1]
            
        candidates.append(candidate)

    weighted = weighted_non_max_suppression(candidates, nms_threshold, max_hands)
    detections = []
    
    for candidate in weighted:
        x1 = model_to_frame_x(candidate.box[0])
        y1 = model_to_frame_y(candidate.box[1])
        x2 = model_to_frame_x(candidate.box[0] + candidate.box[2])
        y2 = model_to_frame_y(candidate.box[1] + candidate.box[3])
        
        frame_box = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
        if frame_box[2] < 2.0 or frame_box[3] < 2.0:
            continue
            
        det = PalmDetection(
            box=[
                max(0.0, min(float(frame_size[0] - 1), frame_box[0])),
                max(0.0, min(float(frame_size[1] - 1), frame_box[1])),
                0.0, 0.0
            ],
            keypoints=[[0.0, 0.0] for _ in range(PALM_KEYPOINT_COUNT)],
            score=candidate.score
        )
        
        right = max(0.0, min(float(frame_size[0] - 1), frame_box[0] + frame_box[2]))
        bottom = max(0.0, min(float(frame_size[1] - 1), frame_box[1] + frame_box[3]))
        det.box[2] = max(0.0, right - det.box[0])
        det.box[3] = max(0.0, bottom - det.box[1])
        
        for k in range(PALM_KEYPOINT_COUNT):
            det.keypoints[k][0] = model_to_frame_x(candidate.keypoints[k][0])
            det.keypoints[k][1] = model_to_frame_y(candidate.keypoints[k][1])
            
        det.hand_roi = make_hand_roi(frame_box, det.keypoints, frame_size)
        detections.append(det)
        
    return detections

# ==============================================================================
# Hand Landmark
# ==============================================================================
def make_landmark_input(frame, roi):
    src = np.array(hand_roi_corners(roi, (frame.shape[1], frame.shape[0])), dtype=np.float32)
    dst = np.array([
        [0.0, 0.0],
        [float(LANDMARK_WIDTH - 1), 0.0],
        [float(LANDMARK_WIDTH - 1), float(LANDMARK_HEIGHT - 1)],
        [0.0, float(LANDMARK_HEIGHT - 1)]
    ], dtype=np.float32)
    
    frame_to_crop = cv2.getPerspectiveTransform(src, dst)
    crop_to_frame = cv2.getPerspectiveTransform(dst, src)
    
    crop_bgr = cv2.warpPerspective(frame, frame_to_crop, (LANDMARK_WIDTH, LANDMARK_HEIGHT), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(crop_rgb, dtype=np.uint8), crop_to_frame

def parse_landmark_outputs(landmarks_tensor, presence_tensor, handedness_tensor, world_tensor, palm, crop_to_frame):
    landmarks = landmarks_tensor.flatten()
    presence_score = presence_tensor.flatten()[0]
    handedness_score = handedness_tensor.flatten()[0]
    
    world = world_tensor.flatten() if world_tensor is not None else None
    
    result = HandLandmarkResult(
        palm=palm,
        landmarks=[],
        world_landmarks=[],
        handedness="Right" if handedness_score > 0.5 else "Left",
        handedness_score=handedness_score,
        confidence=presence_score
    )
    
    crop_points = []
    for i in range(HAND_LANDMARK_COUNT):
        offset = i * 3
        crop_points.append([landmarks[offset], landmarks[offset + 1]])
        
    crop_points_np = np.array([crop_points], dtype=np.float32)
    frame_points = cv2.perspectiveTransform(crop_points_np, crop_to_frame)[0]
    
    for i in range(HAND_LANDMARK_COUNT):
        offset = i * 3
        z = landmarks[offset + 2]
        result.landmarks.append((frame_points[i][0], frame_points[i][1], z))
        if world is not None:
            result.world_landmarks.append((world[offset], world[offset+1], world[offset+2]))
            
    return result

# ==============================================================================
# YOLO Pose
# ==============================================================================
def make_pose_input(frame, input_width, input_height, prep):
    prep.original_width = frame.shape[1]
    prep.original_height = frame.shape[0]
    prep.input_width = input_width
    prep.input_height = input_height
    prep.scale = min(float(input_width) / float(frame.shape[1]), float(input_height) / float(frame.shape[0]))
    
    resized_width = max(1, int(round(float(frame.shape[1]) * prep.scale)))
    resized_height = max(1, int(round(float(frame.shape[0]) * prep.scale)))
    prep.pad_x = (input_width - resized_width) // 2
    prep.pad_y = (input_height - resized_height) // 2
    
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    input_img = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    input_img[prep.pad_y:prep.pad_y+resized_height, prep.pad_x:prep.pad_x+resized_width] = rgb
    
    return np.ascontiguousarray(input_img, dtype=np.uint8)

def parse_pose_outputs(tensor, prep, score_threshold, nms_threshold, num_keypoints=17):
    # Depending on model output shape [1, 22, 8400] or [1, 8400, 22]
    shape = tensor.shape
    if len(shape) < 2: return []
    
    needs_transpose = False
    if len(shape) == 3:
        if shape[1] <= shape[2]:
            channels, count = shape[1], shape[2]
            tensor = np.transpose(tensor[0], (1, 0))
            needs_transpose = True
        else:
            count, channels = shape[1], shape[2]
            tensor = tensor[0]
            needs_transpose = False
    else:
        if shape[0] <= shape[1]:
            channels, count = shape[0], shape[1]
            tensor = np.transpose(tensor, (1, 0))
            needs_transpose = True
        else:
            count, channels = shape[0], shape[1]
            needs_transpose = False
            
    # Assuming YOLOv8/YOLO26s pose format: cx, cy, w, h, score, kpx, kpy, kpscore...
    boxes = []
    scores = []
    indices = []
    
    for i in range(count):
        row = tensor[i]
        score = row[4]
        if score < score_threshold:
            continue
            
        if needs_transpose:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x = cx - w * 0.5
            y = cy - h * 0.5
        else:
            x = row[0]
            y = row[1]
            w = row[2] - row[0]
            h = row[3] - row[1]
            
        if w <= 1.0 or h <= 1.0:
            continue
            
        boxes.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
        scores.append(float(score))
        indices.append(i)
        
    if not boxes: return []
    
    keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
    if len(keep) > 0:
        keep = keep.flatten()
    
    results = []
    for idx in keep:
        row_idx = indices[idx]
        row = tensor[row_idx]
        
        if needs_transpose:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            x1, y1 = cx - w * 0.5, cy - h * 0.5
            x2, y2 = cx + w * 0.5, cy + h * 0.5
        else:
            x1, y1 = row[0], row[1]
            x2, y2 = row[2], row[3]
        
        def scale_x(v): return np.clip((v - prep.pad_x) / prep.scale, 0.0, prep.original_width - 1)
        def scale_y(v): return np.clip((v - prep.pad_y) / prep.scale, 0.0, prep.original_height - 1)
        
        pose = PoseResult(
            box=[scale_x(x1), scale_y(y1), scale_x(x2), scale_y(y2)],
            confidence=scores[idx],
            keypoints=[]
        )
        
        compact_channels = 5 + num_keypoints * 3
        kp_offset = 6 if (not needs_transpose and channels >= compact_channels + 1) else 5
        if channels >= kp_offset + num_keypoints * 3:
            for kp in range(num_keypoints):
                kpx = row[kp_offset + kp * 3]
                kpy = row[kp_offset + kp * 3 + 1]
                kp_conf = row[kp_offset + kp * 3 + 2]
                pose.keypoints.append(PoseKeypoint(scale_x(kpx), scale_y(kpy), kp_conf))
                
        results.append(pose)
        
    results.sort(key=lambda x: x.confidence, reverse=True)
    return results

# ==============================================================================
# Application Threads
# ==============================================================================
def capture_loop(options, cap, queues, running):
    next_frame_id = 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0.0:
        fps = 30.0
    frame_interval = 1.0 / fps
    start_time = time.time()
    
    while running[0]:
        ret, frame = cap.read()
        if not ret or frame is None:
            if not options.use_camera and options.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_time = time.time()
                next_frame_id = 1
                continue
            break
            
        if not options.use_camera:
            elapsed = time.time() - start_time
            expected = next_frame_id * frame_interval
            if elapsed < expected:
                time.sleep(expected - elapsed)
            
        # Crop wide frame to 4:3
        if frame.shape[1] * 3 > frame.shape[0] * 4:
            target_width = max(1, frame.shape[0] * 4 // 3)
            x = max(0, (frame.shape[1] - target_width) // 2)
            frame = frame[:, x:x+target_width]
            
        if options.use_camera:
            frame = cv2.flip(frame, 1)
            
        packet = FramePacket(id=next_frame_id, frame=frame.copy(), captured_at=time.time())
        next_frame_id += 1
        
        queues['capture'].push(packet)
    queues['capture'].close()

def dispatch_loop(options, queues, running):
    while True:
        if not running[0] and queues['capture'].is_closed():
            break
            
        packet = queues['capture'].pop_for(timeout=0.02)
        if packet is None:
            continue
            
        if options.show_pose:
            queues['pose'].push(FramePacket(id=packet.id, frame=packet.frame.copy(), captured_at=packet.captured_at))
        queues['hand'].push(FramePacket(id=packet.id, frame=packet.frame.copy(), captured_at=packet.captured_at))
        queues['render'].push(packet)
        
    queues['pose'].close()
    queues['hand'].close()
    queues['render'].close()

def pose_loop(options, pose_queue, results, running):
    try:
        engine = dx_engine.InferenceEngine(options.pose_model)
        info = engine.get_input_tensors_info()[0]
        shape = info['shape']
        if shape[3] == 3:
            input_height, input_width = shape[1], shape[2]
        else:
            input_height, input_width = shape[2], shape[3]
        while True:
            if not running[0] and pose_queue.is_closed():
                break
                
            packet = pose_queue.pop_for(timeout=0.02)
            if packet is None:
                continue
                
            prep = PosePreprocessInfo()
            input_data = make_pose_input(packet.frame, input_width, input_height, prep)
            
            start_time = time.time()
            outputs = engine.run([input_data])
            inference_ms = (time.time() - start_time) * 1000.0
            
            poses = parse_pose_outputs(outputs[0], prep, options.pose_conf, options.pose_nms, 17)
            results.set_pose(PosePacket(frame_id=packet.id, poses=poses, inference_ms=inference_ms))
            
    except Exception as e:
        print(f"Pose error: {e}")
        running[0] = False

def hand_loop(options, hand_queue, results, running):
    try:
        palm_engine = dx_engine.InferenceEngine(options.palm_model)
        landmark_engine = dx_engine.InferenceEngine(options.landmark_model)
        anchors = generate_palm_anchors()
        
        while True:
            if not running[0] and hand_queue.is_closed():
                break
                
            packet = hand_queue.pop_for(timeout=0.02)
            if packet is None:
                continue
                
            prep = PreprocessInfo()
            palm_input = preprocess_palm_frame(packet.frame, options.keep_aspect, prep)
            
            palm_outputs = palm_engine.run([palm_input])
            
            # Find box and score tensors based on shape
            box_tensor, score_tensor = None, None
            for out in palm_outputs:
                if out.shape[-1] == 18:
                    box_tensor = out
                elif out.shape[-1] == 1:
                    score_tensor = out
                    
            palms = decode_palm_detections(box_tensor, score_tensor, anchors, prep, 
                                           (packet.frame.shape[1], packet.frame.shape[0]), 
                                           options.palm_conf, options.nms, options.max_hands)
            
            hands = []
            if palms:
                # To be efficient in python, we run landmark model synchronously here
                # (since multiple hands will just be batched/looped)
                for palm in palms:
                    lm_input, crop_to_frame = make_landmark_input(packet.frame, palm.hand_roi)
                    lm_outputs = landmark_engine.run([lm_input])
                    # Landmarks output parsing
                    if len(lm_outputs) >= 3:
                        res = parse_landmark_outputs(lm_outputs[0], lm_outputs[1], lm_outputs[2], 
                                                     lm_outputs[3] if len(lm_outputs) > 3 else None,
                                                     palm, crop_to_frame)
                        if res.confidence >= options.landmark_conf:
                            hands.append(res)
                            
                hands.sort(key=lambda x: x.confidence, reverse=True)
                
            results.set_hand(HandPacket(frame_id=packet.id, palms=palms, hands=hands))
            
    except Exception as e:
        print(f"Hand error: {e}")
        running[0] = False

# ==============================================================================
# Rendering & PyQt UI
# ==============================================================================
def draw_results(frame, palms, hands, poses, options):
    base = max(1, min(frame.shape[1], frame.shape[0]))
    box_thickness = max(2, base // 280)
    
    # Draw Poses (Simplified for brevity, but mimicking colors)
    limb_colors = [
        (255, 220, 72), (255, 220, 72), (255, 220, 72), (255, 220, 72),
        (232, 82, 255), (232, 82, 255), (232, 82, 255),
        (255, 226, 48), (255, 226, 48), (255, 226, 48), (255, 226, 48), (255, 226, 48),
        (132, 255, 92), (132, 255, 92), (132, 255, 92), (132, 255, 92), (132, 255, 92), (132, 255, 92), (132, 255, 92)
    ]
    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ]
    
    for pose in poses:
        if len(pose.box) >= 4:
            x1, y1, x2, y2 = pose.box
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (132, 255, 92), box_thickness, cv2.LINE_AA)
            cv2.putText(frame, f"Person: {pose.confidence:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (248, 244, 238), 1, cv2.LINE_AA)
            
        for idx, (a_idx, b_idx) in enumerate(skeleton):
            if a_idx < len(pose.keypoints) and b_idx < len(pose.keypoints):
                kp1, kp2 = pose.keypoints[a_idx], pose.keypoints[b_idx]
                if kp1.confidence > 0.3 and kp2.confidence > 0.3:
                    cv2.line(frame, (int(kp1.x), int(kp1.y)), (int(kp2.x), int(kp2.y)), limb_colors[idx], box_thickness, cv2.LINE_AA)
                    
        for kp in pose.keypoints:
            if kp.confidence > 0.3:
                cv2.circle(frame, (int(kp.x), int(kp.y)), 3, (255, 255, 255), -1, cv2.LINE_AA)

    # Draw Palms & Hands
    if options.show_palm:
        for palm in palms:
            x, y, w, h = map(int, palm.box)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 220, 48), box_thickness, cv2.LINE_AA)
            corners = hand_roi_corners(palm.hand_roi, (frame.shape[1], frame.shape[0]))
            for i in range(len(corners)):
                p1 = (int(corners[i][0]), int(corners[i][1]))
                p2 = (int(corners[(i+1)%len(corners)][0]), int(corners[(i+1)%len(corners)][1]))
                cv2.line(frame, p1, p2, (70, 160, 255), max(1, box_thickness - 1), cv2.LINE_AA)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
        (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14),
        (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    for hand in hands:
        # Draw skeleton
        for (a, b) in connections:
            p1 = (int(hand.landmarks[a][0]), int(hand.landmarks[a][1]))
            p2 = (int(hand.landmarks[b][0]), int(hand.landmarks[b][1]))
            cv2.line(frame, p1, p2, (250, 248, 246), 2, cv2.LINE_AA)
            
        # Draw neon points
        for i, lm in enumerate(hand.landmarks):
            center = (int(lm[0]), int(lm[1]))
            cv2.circle(frame, center, 4, (0, 255, 0), -1, cv2.LINE_AA) # simplified point
            
        cv2.putText(frame, f"{hand.handedness} {hand.confidence:.2f}", (int(hand.landmarks[0][0]), int(hand.landmarks[0][1]) - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 244, 238), 1, cv2.LINE_AA)

class Signals(QObject):
    update_frame = pyqtSignal(QImage, dict)

class FrameView(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEEPX M1 Hand Landmark + Pose Detection")
        self.setFocusPolicy(Qt.StrongFocus)
        self.frame = QImage()
        self.metrics = {}
        self.show_exit_button = False
        self.signals = Signals()
        self.signals.update_frame.connect(self.set_frame)

    def set_exit_button_visible(self, visible):
        self.show_exit_button = visible
        self.update()

    def set_frame(self, image, metrics):
        self.frame = image
        self.metrics = metrics
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if not self.frame.isNull():
            target_size = self.frame.size()
            target_size.scale(self.size(), Qt.KeepAspectRatio)
            top_left = QPoint(0, (self.height() - target_size.height()) // 2)
            painter.drawImage(QRect(top_left, target_size), self.frame)

            panel_rect = QRect(target_size.width(), 0, max(0, self.width() - target_size.width()), self.height())
            self.draw_side_panel(painter, panel_rect)

        self.draw_exit_button(painter)

    def draw_exit_button(self, painter):
        if not self.show_exit_button: return
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = QRect(self.width() - 32 - 14, 14, 32, 28)
        painter.setPen(QPen(QColor(60, 60, 60, 180), 1))
        painter.setBrush(QColor(45, 45, 48, 145))
        painter.drawRoundedRect(r, 6, 6)
        
        font = painter.font()
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(204, 204, 204, 170))
        painter.drawText(r, Qt.AlignCenter, "X")

    def draw_metric_card(self, painter, rect, label, value, accent):
        if rect.width() <= 0 or rect.height() <= 0: return
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(17, 24, 34, 232))
        painter.drawRoundedRect(rect, 8, 8)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRect(rect.x(), rect.y(), 4, rect.height()), 2, 2)
        
        font = painter.font()
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(152, 164, 180))
        painter.drawText(rect.adjusted(18, 12, -16, 0), Qt.AlignLeft | Qt.AlignTop, label)
        
        font.setPixelSize(30)
        painter.setFont(font)
        painter.setPen(QColor(244, 248, 252))
        painter.drawText(rect.adjusted(18, 0, -16, -14), Qt.AlignLeft | Qt.AlignBottom, value)

    def draw_side_panel(self, painter, panel_rect):
        if panel_rect.width() < 180 or panel_rect.height() < 360: return
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        bg = QLinearGradient(panel_rect.topLeft(), panel_rect.bottomRight())
        bg.setColorAt(0.0, QColor(6, 10, 17))
        bg.setColorAt(0.55, QColor(12, 18, 27))
        bg.setColorAt(1.0, QColor(7, 13, 20))
        painter.fillRect(panel_rect, bg)
        
        painter.setPen(QPen(QColor(42, 236, 255, 150), 1))
        painter.drawLine(panel_rect.topLeft(), panel_rect.bottomLeft())
        
        margin = max(24, min(42, panel_rect.width() // 10))
        content = panel_rect.adjusted(margin, 42, -margin, -36)
        y = content.top()
        
        font = painter.font()
        font.setPixelSize(28)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(69, 241, 255))
        painter.drawText(QRect(content.left(), y, content.width(), 28), Qt.AlignLeft | Qt.AlignVCenter, "DEEPX M1")
        y += 32
        
        font.setPixelSize(42)
        painter.setFont(font)
        painter.setPen(QColor(245, 248, 252))
        painter.drawText(QRect(content.left(), y, content.width(), 42), Qt.AlignLeft | Qt.AlignVCenter, "HANDS + POSE")
        y += 50
        
        font.setPixelSize(20)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(154, 166, 180))
        painter.drawText(QRect(content.left(), y, content.width(), 22), Qt.AlignLeft | Qt.AlignVCenter, "21 hand points + YOLO26s pose")
        y += 42
        
        painter.setPen(QPen(QColor(255, 255, 255, 42), 1))
        painter.drawLine(content.left(), y, content.right(), y)
        y += 32
        
        gap = 14
        card_h = max(72, min(104, (content.bottom() - y - 86 - gap * 2) // 3))
        
        self.draw_metric_card(painter, QRect(content.left(), y, content.width(), card_h), "HANDS", 
                              f"{self.metrics.get('hand_count', 0)}/{self.metrics.get('max_hands', 0)}", QColor(83, 255, 134))
        y += card_h + gap
        self.draw_metric_card(painter, QRect(content.left(), y, content.width(), card_h), "PERSONS", 
                              str(self.metrics.get('pose_count', 0)), QColor(255, 217, 74))
        y += card_h + gap
        self.draw_metric_card(painter, QRect(content.left(), y, content.width(), card_h), "FPS", 
                              f"{self.metrics.get('fps', 0.0):.1f}", QColor(47, 221, 255))

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            QApplication.quit()
        elif event.key() == Qt.Key_F:
            if self.isFullScreen(): self.showNormal()
            else: self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        if self.show_exit_button and event.button() == Qt.LeftButton:
            r = QRect(self.width() - 32 - 14, 14, 32, 28)
            if r.contains(event.pos()):
                event.accept()
                QApplication.quit()
                return
        super().mousePressEvent(event)

def render_loop(options, view, render_queue, results, running):
    fps_counter = FpsCounter()
    
    while True:
        if not running[0] and render_queue.is_closed():
            break
            
        packet = render_queue.pop_for(timeout=0.02)
        if packet is None:
            continue
            
        bundle = results.wait_for(packet.id, options.show_pose, True, 0.05)
        
        palms, hands, poses = [], [], []
        if bundle.has_hand and bundle.hand:
            palms = bundle.hand.palms
            hands = bundle.hand.hands
        if options.show_pose and bundle.has_pose and bundle.pose:
            poses = bundle.pose.poses
            
        fps = fps_counter.update()
        metrics = {
            'hand_count': len(hands),
            'palm_count': len(palms),
            'pose_count': len(poses),
            'fps': fps,
            'max_hands': options.max_hands,
        }
        
        draw_results(packet.frame, palms, hands, poses, options)
        
        rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
        view.signals.update_frame.emit(qimage, metrics)
        
        keep_from = packet.id - 90 if packet.id > 90 else 0
        results.prune_before(keep_from)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--palm-model', default=DEFAULT_PALM_MODEL)
    parser.add_argument('--landmark-model', default=DEFAULT_LANDMARK_MODEL)
    parser.add_argument('--pose-model', default=DEFAULT_POSE_MODEL)
    parser.add_argument('-c', '--camera', type=int, default=0)
    parser.add_argument('-v', '--video', type=str, default='')
    parser.add_argument('--max-hands', type=int, default=8)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--palm-conf', type=float, default=0.2)
    parser.add_argument('--landmark-conf', type=float, default=0.5)
    parser.add_argument('--pose-conf', type=float, default=0.3)
    parser.add_argument('--nms', type=float, default=0.3)
    parser.add_argument('--pose-nms', type=float, default=0.45)
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--stretch', action='store_true')
    parser.add_argument('--hide-palm', action='store_true')
    parser.add_argument('--hide-pose', action='store_true')
    parser.add_argument('--exit-btn', action='store_true')
    parser.add_argument('--windowed', action='store_true')
    parser.add_argument('--fullscreen', action='store_true', default=True)
    args = parser.parse_args()

    args.keep_aspect = not args.stretch
    args.show_palm = not args.hide_palm
    args.show_pose = not args.hide_pose
    args.use_camera = not bool(args.video)

    app = QApplication(sys.argv)
    view = FrameView()
    view.set_exit_button_visible(args.exit_btn)
    
    if args.windowed: args.fullscreen = False
    
    if args.fullscreen:
        view.showFullScreen()
    else:
        view.resize(1280, 720)
        view.show()

    cap = cv2.VideoCapture(args.camera if args.use_camera else args.video)
    if args.use_camera:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        
    queues = {
        'capture': BoundedQueue(QUEUE_SIZE),
        'pose': BoundedQueue(QUEUE_SIZE),
        'hand': BoundedQueue(QUEUE_SIZE),
        'render': BoundedQueue(QUEUE_SIZE + 1)
    }
    
    results = ResultStore()
    running = [True]

    threads = [
        threading.Thread(target=capture_loop, args=(args, cap, queues, running)),
        threading.Thread(target=dispatch_loop, args=(args, queues, running)),
        threading.Thread(target=hand_loop, args=(args, queues['hand'], results, running)),
        threading.Thread(target=render_loop, args=(args, view, queues['render'], results, running))
    ]
    if args.show_pose:
        threads.append(threading.Thread(target=pose_loop, args=(args, queues['pose'], results, running)))

    for t in threads:
        t.start()

    rc = app.exec_()
    running[0] = False
    
    for q in queues.values():
        q.close()
        
    for t in threads:
        t.join()
        
    sys.exit(rc)

if __name__ == '__main__':
    main()
