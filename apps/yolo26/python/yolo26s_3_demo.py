import sys
import os
import argparse
import time
import math
from pathlib import Path
from collections import deque
import threading
import queue

import cv2
import numpy as np

from dx_engine import InferenceEngine

from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QGridLayout, QSizePolicy, QMainWindow
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QFont, QColor
from PySide6.QtCore import Qt, QTimer, Signal, QThread

def make_letterbox(img, target_w=640, target_h=640, pad_value=114):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return padded, scale, pad_w, pad_h

class BoundedQueue:
    def __init__(self, max_size):
        self.max_size = max_size
        self.items = deque()
        self.cond = threading.Condition()
        self.closed = False

    def push(self, item):
        with self.cond:
            if self.closed: return False
            while len(self.items) >= self.max_size:
                self.items.popleft()
            self.items.append(item)
            self.cond.notify()
        return True

    def pop_for(self, timeout):
        with self.cond:
            if not self.items and not self.closed:
                self.cond.wait(timeout)
            if not self.items: return None
            return self.items.popleft()

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()

class FramePacket:
    def __init__(self, id, frame, raw_frame):
        self.id = id
        self.frame = frame
        self.raw_frame = raw_frame

class ResultBundle:
    def __init__(self, frame_id, raw_frame):
        self.frame_id = frame_id
        self.raw_frame = raw_frame
        self.od = None
        self.pose = None
        self.seg = None

class ResultStore:
    def __init__(self):
        self.cond = threading.Condition()
        self.od_results = {}
        self.pose_results = {}
        self.seg_results = {}
        
        self.latest_od = None
        self.latest_pose = None
        self.latest_seg = None
        
        self.MAX_CACHE = 10

    def add_od(self, frame_id, result):
        with self.cond:
            self.od_results[frame_id] = result
            self.latest_od = result
            self._cleanup()
            self.cond.notify_all()

    def add_pose(self, frame_id, result):
        with self.cond:
            self.pose_results[frame_id] = result
            self.latest_pose = result
            self._cleanup()
            self.cond.notify_all()

    def add_seg(self, frame_id, result):
        with self.cond:
            self.seg_results[frame_id] = result
            self.latest_seg = result
            self._cleanup()
            self.cond.notify_all()

    def _cleanup(self):
        if len(self.od_results) > self.MAX_CACHE:
            oldest = min(self.od_results.keys())
            del self.od_results[oldest]
        if len(self.pose_results) > self.MAX_CACHE:
            oldest = min(self.pose_results.keys())
            del self.pose_results[oldest]
        if len(self.seg_results) > self.MAX_CACHE:
            oldest = min(self.seg_results.keys())
            del self.seg_results[oldest]

    def wait_for(self, frame_id, timeout=0.1):
        start_time = time.time()
        bundle = ResultBundle(frame_id, None)
        with self.cond:
            while True:
                ready = True
                if frame_id not in self.od_results: ready = False
                if frame_id not in self.pose_results: ready = False
                if frame_id not in self.seg_results: ready = False
                
                if ready:
                    bundle.od = self.od_results[frame_id]
                    bundle.pose = self.pose_results[frame_id]
                    bundle.seg = self.seg_results[frame_id]
                    return bundle
                
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    bundle.od = self.od_results.get(frame_id, self.latest_od)
                    bundle.pose = self.pose_results.get(frame_id, self.latest_pose)
                    bundle.seg = self.seg_results.get(frame_id, self.latest_seg)
                    return bundle
                
                self.cond.wait(timeout - elapsed)

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

class ODWorker(threading.Thread):
    def __init__(self, model_path, in_queue, store, running):
        super().__init__()
        self.model_path = model_path
        self.in_queue = in_queue
        self.store = store
        self.running = running
        self.engine = None
        
    def run(self):
        try:
            self.engine = InferenceEngine(self.model_path)
        except Exception as e:
            print(f"Failed to load OD model: {e}")
            return
            
        while self.running[0]:
            packet = self.in_queue.pop_for(0.1)
            if not packet: continue
            
            padded, scale, pad_w, pad_h = make_letterbox(packet.frame)
            img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            input_tensor = np.ascontiguousarray(img_rgb, dtype=np.uint8)
            
            job_id = self.engine.run_async([input_tensor])
            outputs = self.engine.wait(job_id)
            if not outputs: continue
            
            # Post-processing [1, 300, 6]
            out = outputs[0]
            if len(out.shape) == 3: out = out[0]
            
            results = []
            for row in out:
                if row[4] < 0.25: continue # score threshold
                x1, y1, x2, y2 = row[0:4]
                score = row[4]
                cls_id = int(round(row[5]))
                
                # unpad and scale
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale
                
                results.append({
                    "box": [x1, y1, x2, y2],
                    "score": score,
                    "class_id": cls_id
                })
                
            self.store.add_od(packet.id, results)

class PoseWorker(threading.Thread):
    def __init__(self, model_path, in_queue, store, running):
        super().__init__()
        self.model_path = model_path
        self.in_queue = in_queue
        self.store = store
        self.running = running
        self.engine = None
        
    def run(self):
        try:
            self.engine = InferenceEngine(self.model_path)
        except Exception as e:
            print(f"Failed to load Pose model: {e}")
            return
            
        while self.running[0]:
            packet = self.in_queue.pop_for(0.1)
            if not packet: continue
            
            padded, scale, pad_w, pad_h = make_letterbox(packet.frame)
            img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            input_tensor = np.ascontiguousarray(img_rgb, dtype=np.uint8)
            
            job_id = self.engine.run_async([input_tensor])
            outputs = self.engine.wait(job_id)
            if not outputs: continue
            
            out = outputs[0]
            shape = out.shape
            needs_transpose = False
            if len(shape) == 3:
                if shape[1] <= shape[2]:
                    out = np.transpose(out[0], (1, 0))
                    needs_transpose = True
                else:
                    out = out[0]
            else:
                if shape[0] <= shape[1]:
                    out = np.transpose(out, (1, 0))
                    needs_transpose = True
                    
            boxes = []
            scores = []
            keypoints_list = []
            
            for row in out:
                score = row[4]
                if score < 0.25: continue
                
                if needs_transpose:
                    cx, cy, w, h = row[0:4]
                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2
                    kp_offset = 5
                else:
                    x1, y1, x2, y2 = row[0:4]
                    kp_offset = 6
                    
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                
                kps = []
                for k in range(17):
                    kx = row[kp_offset + k*3]
                    ky = row[kp_offset + k*3 + 1]
                    kconf = row[kp_offset + k*3 + 2]
                    kps.append([kx, ky, kconf])
                keypoints_list.append(kps)
                
            keep = nms(boxes, scores, 0.45)
            
            results = []
            for k in keep:
                x1, y1, x2, y2 = boxes[k]
                x1 = (x1 - pad_w) / scale
                y1 = (y1 - pad_h) / scale
                x2 = (x2 - pad_w) / scale
                y2 = (y2 - pad_h) / scale
                
                kps = keypoints_list[k]
                scaled_kps = []
                for kx, ky, kconf in kps:
                    sx = (kx - pad_w) / scale
                    sy = (ky - pad_h) / scale
                    scaled_kps.append([sx, sy, kconf])
                    
                results.append({
                    "box": [x1, y1, x2, y2],
                    "score": scores[k],
                    "keypoints": scaled_kps
                })
                
            self.store.add_pose(packet.id, results)

class SegWorker(threading.Thread):
    def __init__(self, model_path, in_queue, store, running):
        super().__init__()
        self.model_path = model_path
        self.in_queue = in_queue
        self.store = store
        self.running = running
        self.engine = None
        
    def run(self):
        try:
            self.engine = InferenceEngine(self.model_path)
        except Exception as e:
            print(f"Failed to load Seg model: {e}")
            return
            
        while self.running[0]:
            packet = self.in_queue.pop_for(0.1)
            if not packet: continue
            
            orig_h, orig_w = packet.frame.shape[:2]
            padded, scale, pad_w, pad_h = make_letterbox(packet.frame)
            img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            input_tensor = np.ascontiguousarray(img_rgb, dtype=np.uint8)
            
            job_id = self.engine.run_async([input_tensor])
            outputs = self.engine.wait(job_id)
            if not outputs or len(outputs) < 2: continue
            
            det_out = outputs[0]
            mask_proto = outputs[1]
            
            if len(det_out.shape) == 3: det_out = det_out[0]
            if len(mask_proto.shape) == 4: mask_proto = mask_proto[0] # [32, 160, 160]
            
            proto_c, proto_h, proto_w = mask_proto.shape
            mask_proto_flat = mask_proto.reshape(proto_c, -1)
            
            results = []
            for row in det_out:
                score = row[4]
                if score < 0.25: continue
                x1, y1, x2, y2 = row[0:4]
                cls_id = int(round(row[5]))
                coefs = row[6:6+proto_c]
                
                bx1, by1 = max(0, int(round(x1/4))), max(0, int(round(y1/4)))
                bx2, by2 = min(proto_w, int(round(x2/4))), min(proto_h, int(round(y2/4)))
                
                if bx2 > bx1 and by2 > by1:
                    mask_proto_roi = mask_proto[:, by1:by2, bx1:bx2]
                    roi_flat = mask_proto_roi.reshape(proto_c, -1)
                    mask_logits = np.dot(coefs, roi_flat)
                    mask_logits = mask_logits.reshape(by2-by1, bx2-bx1)
                    box_mask = 1.0 / (1.0 + np.exp(-mask_logits))
                    box_mask_binary = (box_mask > 0.5).astype(np.uint8) * 255
                else:
                    box_mask_binary = np.zeros((1, 1), dtype=np.uint8)
                
                ux1 = (x1 - pad_w) / scale
                uy1 = (y1 - pad_h) / scale
                ux2 = (x2 - pad_w) / scale
                uy2 = (y2 - pad_h) / scale
                
                results.append({
                    "box": [ux1, uy1, ux2, uy2],
                    "score": score,
                    "class_id": cls_id,
                    "mask_roi": box_mask_binary
                })
                
            self.store.add_seg(packet.id, results)

def draw_od(frame, results):
    if not results: return frame
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), r["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"cls:{r['class_id']} {r['score']:.2f}"
        cv2.putText(out, label, (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out

def draw_pose(frame, results):
    if not results: return frame
    out = frame.copy()
    
    skeleton = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]]
                
    for r in results:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), r["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        kps = r["keypoints"]
        for pt in kps:
            kx, ky, kconf = pt
            if kconf > 0.5:
                cv2.circle(out, (int(round(kx)), int(round(ky))), 4, (0, 0, 255), -1)
                
        for sk in skeleton:
            idx1, idx2 = sk
            if idx1 < len(kps) and idx2 < len(kps):
                pt1, pt2 = kps[idx1], kps[idx2]
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    p1 = (int(round(pt1[0])), int(round(pt1[1])))
                    p2 = (int(round(pt2[0])), int(round(pt2[1])))
                    cv2.line(out, p1, p2, (0, 255, 255), 2)
    return out

def draw_seg(frame, results):
    if not results: return frame
    out = frame.copy()
    
    for r in results:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), r["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        bw = x2 - x1
        bh = y2 - y1
        if bw > 0 and bh > 0:
            mask_roi = r["mask_roi"]
            if mask_roi.shape[0] == 0 or mask_roi.shape[1] == 0: continue
            
            resized_mask = cv2.resize(mask_roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
            
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(out.shape[1], x2), min(out.shape[0], y2)
            bw_c = x2_c - x1_c
            bh_c = y2_c - y1_c
            
            if bw_c > 0 and bh_c > 0:
                mx1 = x1_c - x1
                my1 = y1_c - y1
                mx2 = mx1 + bw_c
                my2 = my1 + bh_c
                
                cropped_mask = resized_mask[my1:my2, mx1:mx2]
                mask_indices = cropped_mask > 127
                
                roi = out[y1_c:y2_c, x1_c:x2_c]
                colored_mask = np.zeros_like(roi)
                colored_mask[:] = (0, 128, 255)
                
                roi[mask_indices] = roi[mask_indices] * 0.5 + colored_mask[mask_indices] * 0.5
                out[y1_c:y2_c, x1_c:x2_c] = roi
                
    return out

class VideoThread(QThread):
    frame_ready = Signal(object)
    def __init__(self, queues, store, running, options):
        super().__init__()
        self.queues = queues
        self.store = store
        self.running = running
        self.options = options
        
    def run(self):
        source = 0 if self.options.video == "" else self.options.video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Failed to open video source")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        frame_interval = 1.0 / fps
        
        frame_id = 0
        start_time = time.time()
        
        while self.running[0]:
            ret, frame = cap.read()
            if not ret:
                if self.options.video != "" and not self.options.no_loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_time = time.time()
                    frame_id = 0
                    continue
                else:
                    break
                    
            if self.options.video != "":
                elapsed = time.time() - start_time
                expected = frame_id * frame_interval
                if elapsed < expected:
                    time.sleep(expected - elapsed)
                    
            frame_id += 1
            raw_frame = frame.copy()
            
            packet = FramePacket(frame_id, frame, raw_frame)
            self.queues['od'].push(FramePacket(frame_id, frame.copy(), None))
            self.queues['pose'].push(FramePacket(frame_id, frame.copy(), None))
            self.queues['seg'].push(FramePacket(frame_id, frame.copy(), None))
            
            bundle = self.store.wait_for(frame_id, timeout=0.0)
            bundle.raw_frame = raw_frame
            self.frame_ready.emit(bundle)
            
        cap.release()

class DemoWindow(QMainWindow):
    def __init__(self, options):
        super().__init__()
        self.setWindowTitle("YOLO26-S 2x2 Demo")
        self.resize(1280, 720)
        self.options = options
        
        main_widget = QWidget()
        layout = QGridLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        self.labels = []
        for i in range(4):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            lbl.setStyleSheet("background-color: black; border: 1px solid gray;")
            layout.addWidget(lbl, i // 2, i % 2)
            self.labels.append(lbl)
            
        self.demo_img = None
        if os.path.exists(options.demo_image):
            self.demo_img = cv2.imread(options.demo_image)
            
        self.running = [True]
        self.store = ResultStore()
        self.queues = {
            'od': BoundedQueue(2),
            'pose': BoundedQueue(2),
            'seg': BoundedQueue(2)
        }
        
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        self.workers = [
            ODWorker(options.model, self.queues['od'], self.store, self.running),
            PoseWorker(options.model_pose, self.queues['pose'], self.store, self.running),
            SegWorker(options.model_seg, self.queues['seg'], self.store, self.running)
        ]
        for w in self.workers: w.start()
        
        self.video_thread = VideoThread(self.queues, self.store, self.running, options)
        self.video_thread.frame_ready.connect(self.update_frames)
        self.video_thread.start()
        
    def update_frames(self, bundle):
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time) if current_time > self.last_time else 0
        self.last_time = current_time
        self.fps_queue.append(fps)
        avg_fps = sum(self.fps_queue) / len(self.fps_queue)
        fps_str = f" | FPS: {avg_fps:.1f}"

        if not bundle.raw_frame is None:
            od_img = draw_od(bundle.raw_frame, bundle.od)
            pose_img = draw_pose(bundle.raw_frame, bundle.pose)
            seg_img = draw_seg(bundle.raw_frame, bundle.seg)
            
            self.set_image(self.labels[0], od_img, "YOLO26-S Object Detection" + fps_str)
            self.set_image(self.labels[1], pose_img, "YOLO26-S Pose Estimation" + fps_str)
            self.set_image(self.labels[2], seg_img, "YOLO26-S Instance Segmentation" + fps_str)
            
            if self.demo_img is not None:
                self.set_image(self.labels[3], self.demo_img, "YOLO26-S Demo")
            else:
                self.set_image(self.labels[3], bundle.raw_frame, "YOLO26-S Demo" + fps_str)
                
    def set_image(self, label, cv_img, title):
        target_w, target_h = label.size().width(), label.size().height()
        if target_w > 0 and target_h > 0:
            cv_img = cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        qimg = QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        pix = QPixmap.fromImage(qimg)
        
        # Draw title
        painter = QPainter(pix)
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        painter.drawText(10, 30, title)
        painter.end()
        
        label.setPixmap(pix)

    def closeEvent(self, event):
        self.running[0] = False
        for q in self.queues.values(): q.close()
        for w in self.workers: w.join()
        self.video_thread.wait()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q or event.key() == Qt.Key.Key_Escape:
            self.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../workspace/models/common/yolo26s.dxnn")
    parser.add_argument("--model-pose", type=str, default="../workspace/models/common/yolo26s-pose.dxnn")
    parser.add_argument("--model-seg", type=str, default="../workspace/models/common/yolo26s-seg.dxnn")
    parser.add_argument("--demo-image", type=str, default="../../assets/yolo26/yolo26-demo.png")
    parser.add_argument("-v", "--video", type=str, default="")
    parser.add_argument("--no-loop-video", action="store_true")
    args = parser.parse_args()
    
    # Notify launcher
    path = os.environ.get("DX_LAUNCHER_READY_FILE")
    if path:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("ready\n", encoding="utf-8")
        except OSError:
            pass
            
    app = QApplication(sys.argv)
    win = DemoWindow(args)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
