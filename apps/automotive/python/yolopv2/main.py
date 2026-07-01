import argparse
import sys
import time
import os

import numpy as np
import cv2

try:
    import dx_engine
    DX_AVAILABLE = True
except ImportError:
    DX_AVAILABLE = False

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QVBoxLayout

import threading
import queue

# YOLOPv2 Constants
NET_W = 640
NET_H = 384
NUM_CLASSES = 80
ANCHORS = [
    [(12.0, 16.0), (19.0, 36.0), (40.0, 28.0)],
    [(36.0, 75.0), (76.0, 55.0), (72.0, 146.0)],
    [(142.0, 110.0), (192.0, 243.0), (459.0, 401.0)]
]
STRIDES = [8, 16, 32]
CONF_THRESH = 0.3
IOU_THRESH = 0.45

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def letterbox(img, new_shape=(NET_H, NET_W), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, dw, dh, top, left)

def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return []
    
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
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
        
    return keep

def decode_boxes(det_outputs, info):
    r, dw, dh, top, left = info
    boxes_list = []
    scores_list = []
    class_list = []
    
    # det_outputs are usually ordered det0, det1, det2 (stride 8, 16, 32)
    # DXNN outputs: [1, 255, 48, 80], [1, 255, 24, 40], [1, 255, 12, 20]
    
    for layer, out in enumerate(det_outputs):
        # out shape: [1, 255, H, W]
        # Reshape to [3, 85, H, W]
        H, W = out.shape[2], out.shape[3]
        stride = STRIDES[layer]
        anchors = ANCHORS[layer]
        
        out = out[0].reshape((3, 85, H, W))
        out = sigmoid(out)
        
        obj = out[:, 4, :, :]
        mask = obj > CONF_THRESH
        
        # Get indices of positive detections
        a_idx, y_idx, x_idx = np.where(mask)
        
        for i in range(len(a_idx)):
            a, gy, gx = a_idx[i], y_idx[i], x_idx[i]
            conf = obj[a, gy, gx]
            
            cls_scores = out[a, 5:, gy, gx]
            cls_id = np.argmax(cls_scores)
            score = conf * cls_scores[cls_id]
            
            if score > CONF_THRESH:
                bx = (out[a, 0, gy, gx] * 2.0 - 0.5 + gx) * stride
                by = (out[a, 1, gy, gx] * 2.0 - 0.5 + gy) * stride
                bw = (out[a, 2, gy, gx] * 2.0) ** 2 * anchors[a][0]
                bh = (out[a, 3, gy, gx] * 2.0) ** 2 * anchors[a][1]
                
                # Transform to original image scale
                bx = (bx - left) / r
                by = (by - top) / r
                bw /= r
                bh /= r
                
                x1 = bx - bw / 2
                y1 = by - bh / 2
                x2 = bx + bw / 2
                y2 = by + bh / 2
                
                boxes_list.append([x1, y1, x2, y2])
                scores_list.append(score)
                class_list.append(cls_id)
                
    if len(boxes_list) == 0:
        return []
        
    boxes = np.array(boxes_list)
    scores = np.array(scores_list)
    classes = np.array(class_list)
    
    keep = nms(boxes, scores, IOU_THRESH)
    
    result = []
    for i in keep:
        result.append({
            'box': boxes[i],
            'score': scores[i],
            'class': classes[i]
        })
    return result

def make_drive_mask(drive_out, info, orig_shape):
    # drive_out: [1, 2, 384, 640]
    r, dw, dh, top, left = info
    
    bg = drive_out[0, 0]
    drive = drive_out[0, 1]
    
    mask = drive > bg
    mask = mask.astype(np.uint8)
    
    # Remove padding
    H, W = orig_shape[:2]
    mask = mask[top:top+int(round(H*r)), left:left+int(round(W*r))]
    
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
    return mask

def make_lane_mask(lane_out, info, orig_shape):
    # lane_out: [1, 1, 384, 640]
    r, dw, dh, top, left = info
    
    lane = lane_out[0, 0]
    mask = lane >= 0.5
    mask = mask.astype(np.uint8)
    
    H, W = orig_shape[:2]
    mask = mask[top:top+int(round(H*r)), left:left+int(round(W*r))]
    
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
    return mask

class YOLOPv2App:
    def __init__(self, model_path, video_path):
        if not DX_AVAILABLE:
            raise RuntimeError("DXNN backend requested but dx_engine is not installed.")
            
        print(f"[Info] Loading DXNN model from: {model_path}")
        self.session = dx_engine.InferenceEngine(model_path)
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
            
        input_info = self.session.get_input_tensors_info()
        self.dtype = input_info[0]['dtype']
        
        self.stop_event = threading.Event()
        self.decode_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=2)
        
        self.job_id_counter = 0
        self.pending_jobs = {}
        self.jobs_lock = threading.Lock()
        
        self.session.register_callback(self._inference_callback)
        
        self.worker_thread = threading.Thread(target=self._data_prep_loop)
        self.decode_thread = threading.Thread(target=self._decode_loop)
        
        self.worker_thread.start()
        self.decode_thread.start()

    def stop(self):
        self.stop_event.set()
        try:
            if hasattr(self.session, 'unregister_callback'):
                self.session.unregister_callback(self._inference_callback)
            elif hasattr(self.session, 'clear_callback'):
                self.session.clear_callback()
        except Exception:
            pass
        self.worker_thread.join()
        self.decode_thread.join()

    def _inference_callback(self, outputs, user_arg):
        if self.stop_event.is_set():
            return 0
        job_id = user_arg
        cloned_outputs = [np.array(o).copy() for o in outputs]
        self.decode_queue.put((job_id, cloned_outputs))
        return 0

    def _data_prep_loop(self):
        while not self.stop_event.is_set():
            with self.jobs_lock:
                num_pending = len(self.pending_jobs)
                
            if num_pending > 2 or self.result_queue.full():
                time.sleep(0.001)
                continue
                
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
            # Resize frame slightly for better display speed in python
            frame = cv2.resize(frame, (1280, 720))
            
            boxed, info = letterbox(frame)
            tensor = boxed.astype(self.dtype)
            
            job_id = self.job_id_counter
            self.job_id_counter += 1
            
            with self.jobs_lock:
                self.pending_jobs[job_id] = {
                    'frame': frame,
                    'info': info,
                    'start_time': start_time
                }
                
            self.session.run_async([tensor], user_arg=job_id)

    def _decode_loop(self):
        out_info = self.session.get_output_tensors_info()
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0.0
        while not self.stop_event.is_set():
            try:
                job_id, outputs = self.decode_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            with self.jobs_lock:
                if job_id not in self.pending_jobs:
                    continue
                job = self.pending_jobs[job_id]
                
            det_outputs = [None, None, None]
            drive_out = None
            lane_out = None
            
            for i, info_t in enumerate(out_info):
                name = info_t['name']
                shape = info_t['shape']
                out_tensor = outputs[i].reshape(shape)
                
                if 'det0' in name or (shape[1] == 255 and shape[2] == 48):
                    det_outputs[0] = out_tensor
                elif 'det1' in name or (shape[1] == 255 and shape[2] == 24):
                    det_outputs[1] = out_tensor
                elif 'det2' in name or (shape[1] == 255 and shape[2] == 12):
                    det_outputs[2] = out_tensor
                elif shape[1] == 2:
                    drive_out = out_tensor
                elif shape[1] == 1:
                    lane_out = out_tensor
                    
            frame = job['frame']
            info = job['info']
            start_time = job['start_time']
            
            boxes = decode_boxes(det_outputs, info)
            drive_mask = make_drive_mask(drive_out, info, frame.shape)
            lane_mask = make_lane_mask(lane_out, info, frame.shape)
            
            res = frame.copy()
            
            drive_color = np.array([92, 238, 118], dtype=np.uint8)
            lane_color = np.array([255, 70, 224], dtype=np.uint8)
            
            res[drive_mask == 1] = res[drive_mask == 1] * 0.5 + drive_color * 0.5
            res[lane_mask == 1] = lane_color
            
            for b in boxes:
                x1, y1, x2, y2 = map(int, b['box'])
                cv2.rectangle(res, (x1, y1), (x2, y2), (255, 230, 64), 2)

            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.avg_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            cv2.putText(res, f"Python Backend - Async FPS: {self.avg_fps:.1f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if not self.result_queue.full():
                self.result_queue.put(res)
                
            with self.jobs_lock:
                del self.pending_jobs[job_id]

    def get_latest_result(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None

class DemoWindow(QMainWindow):
    def __init__(self, app_model, full_screen):
        super().__init__()
        self.app_model = app_model
        
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: black;")
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_img = QLabel()
        self.lbl_img.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_img)
        
        if full_screen:
            self.showFullScreen()
        else:
            self.resize(1280, 720)
            self.show()
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16) # ~60fps UI refresh
        
    def update_frame(self):
        img = self.app_model.get_latest_result()
        if img is not None:
            lbl_w, lbl_h = self.lbl_img.width(), self.lbl_img.height()
            if lbl_w > 0 and lbl_h > 0:
                aspect_ratio = img.shape[1] / img.shape[0]
                if lbl_w / lbl_h > aspect_ratio:
                    new_h = lbl_h
                    new_w = int(new_h * aspect_ratio)
                else:
                    new_w = lbl_w
                    new_h = int(new_w / aspect_ratio)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.lbl_img.setPixmap(QPixmap.fromImage(qimg))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.app_model.stop()
            self.close()

    def closeEvent(self, event):
        self.app_model.stop()
        super().closeEvent(event)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--backend", type=str, default="dxnn")
    parser.add_argument("--full_screen", action="store_true")
    parser.add_argument("--exit-btn", action="store_true")
    args = parser.parse_args()

    app_model = YOLOPv2App(args.model, args.video)
    
    app = QApplication(sys.argv)
    window = DemoWindow(app_model, args.full_screen)
    app.exec_()

if __name__ == "__main__":
    main()
