import sys
import os
import argparse
import time
import math
import json
import threading

import cv2
import numpy as np

from dx_engine import InferenceEngine

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QTimer

def make_letterbox(img, target_w=512, target_h=512, pad_value=114):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    padded = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    return padded, scale, pad_w, pad_h

class MultiWorker(threading.Thread):
    def __init__(self, idx, model_path, source, source_type, num_frames, running, target_w, target_h):
        super().__init__()
        self.idx = idx
        self.model_path = model_path
        self.source = source
        self.source_type = source_type
        self.num_frames = num_frames
        self.running = running
        self.target_w = target_w
        self.target_h = target_h
        
        self.latest_frame = None
        self.lock = threading.Lock()
        self.avg_fps = 0.0
        
        self.preloaded = []
        if self.source_type == "offline" and self.num_frames > 0:
            cap = cv2.VideoCapture(self.source)
            count = 0
            while count < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                padded, scale, pad_w, pad_h = make_letterbox(frame, 512, 512)
                img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                input_tensor = np.ascontiguousarray(img_rgb, dtype=np.uint8)
                self.preloaded.append((frame, input_tensor, scale, pad_w, pad_h))
                count += 1
            cap.release()
        
    def run(self):
        try:
            engine = InferenceEngine(self.model_path)
        except Exception as e:
            print(f"[{self.idx}] Failed to load model: {e}")
            return
            
        cap = None
        if not self.preloaded:
            source_val = 0 if self.source == "/dev/video0" else self.source
            cap = cv2.VideoCapture(source_val)
            if not cap.isOpened():
                print(f"[{self.idx}] Failed to open source {self.source}")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            frame_interval = 1.0 / fps
        else:
            frame_interval = 1.0 / 30.0 # default to 30 fps for preloaded

        frame_id = 0
        start_time = time.time()
        
        fps_history = []
        last_frame_time = start_time
        
        while self.running[0]:
            if self.preloaded:
                idx = frame_id % len(self.preloaded)
                frame, input_tensor, scale, pad_w, pad_h = self.preloaded[idx]
            else:
                ret, frame = cap.read()
                if not ret:
                    if self.source_type == "offline":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        start_time = time.time()
                        frame_id = 0
                        continue
                    else:
                        break
                        
                padded, scale, pad_w, pad_h = make_letterbox(frame, 512, 512)
                img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                input_tensor = np.ascontiguousarray(img_rgb, dtype=np.uint8)
                    
            if self.source_type == "offline":
                elapsed = time.time() - start_time
                expected = frame_id * frame_interval
                if elapsed < expected:
                    time.sleep(expected - elapsed)
                    
            frame_id += 1
            
            job_id = engine.run_async([input_tensor])
            outputs = engine.wait(job_id)
            
            # Drawing
            display_frame = frame.copy()
            if outputs:
                out_raw = outputs[0]
                if len(out_raw.shape) == 3: out_raw = out_raw[0]
                
                out_f32 = out_raw.view(np.float32)
                out_i32 = out_raw.view(np.int32)
                
                if out_f32.ndim >= 2 and out_f32.shape[0] > 0 and out_f32.shape[1] >= 6:
                    scores = out_f32[:, 5]
                    mask = scores >= 0.25
                    
                    if np.any(mask):
                        out_f32_m = out_f32[mask]
                        out_i32_m = out_i32[mask]
                        
                        info = out_i32_m[:, 4]
                        grid_y = info & 0xFF
                        grid_x = (info >> 8) & 0xFF
                        box_idx = (info >> 16) & 0xFF
                        layer_idx = (info >> 24) & 0xFF
                        
                        anchors_np = np.array([
                            [(10,13), (16,30), (33,23)],
                            [(30,61), (62,45), (59,119)],
                            [(116,90), (156,198), (373,326)]
                        ], dtype=np.float32)
                        strides_np = np.array([8, 16, 32], dtype=np.float32)
                        
                        valid = (layer_idx < 3) & (box_idx < 3)
                        if np.any(valid):
                            out_f32_v = out_f32_m[valid]
                            grid_y = grid_y[valid]
                            grid_x = grid_x[valid]
                            box_idx = box_idx[valid]
                            layer_idx = layer_idx[valid]
                            
                            stride = strides_np[layer_idx]
                            anchor = anchors_np[layer_idx, box_idx]
                            anchor_w = anchor[:, 0]
                            anchor_h = anchor[:, 1]
                            
                            x = out_f32_v[:, 0]
                            y = out_f32_v[:, 1]
                            w = out_f32_v[:, 2]
                            h = out_f32_v[:, 3]
                            
                            bx = (x * 2.0 - 0.5 + grid_x) * stride
                            by = (y * 2.0 - 0.5 + grid_y) * stride
                            bw = ((w * 2.0) ** 2) * anchor_w
                            bh = ((h * 2.0) ** 2) * anchor_h
                            
                            x1 = (bx - bw / 2 - pad_w) / scale
                            y1 = (by - bh / 2 - pad_h) / scale
                            x2 = (bx + bw / 2 - pad_w) / scale
                            y2 = (by + bh / 2 - pad_h) / scale
                            
                            x1 = np.round(x1).astype(np.int32)
                            y1 = np.round(y1).astype(np.int32)
                            x2 = np.round(x2).astype(np.int32)
                            y2 = np.round(y2).astype(np.int32)
                            
                            bboxes = []
                            valid_scores = out_f32_v[:, 5]
                            for i in range(len(x1)):
                                bboxes.append([int(x1[i]), int(y1[i]), int(x2[i] - x1[i]), int(y2[i] - y1[i])])
                                
                            if len(bboxes) > 0:
                                indices = cv2.dnn.NMSBoxes(bboxes, valid_scores, 0.25, 0.45)
                                if len(indices) > 0:
                                    for i in indices.flatten():
                                        bx1, by1, bw_box, bh_box = bboxes[i]
                                        bx1 = max(0, bx1)
                                        by1 = max(0, by1)
                                        bx2 = min(frame.shape[1], bx1 + bw_box)
                                        by2 = min(frame.shape[0], by1 + bh_box)
                                        cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    
            now = time.time()
            fps_val = 1.0 / (now - last_frame_time + 1e-6)
            last_frame_time = now
            fps_history.append(fps_val)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            self.avg_fps = avg_fps
            
            caption = f"CH{self.idx + 1} / {avg_fps:.2f} FPS"
            cv2.rectangle(display_frame, (0, 0), (230, 34), (0, 0, 0), cv2.FILLED)
            cv2.putText(display_frame, caption, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            display_frame = cv2.resize(display_frame, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            
            with self.lock:
                self.latest_frame = display_frame
                
        if cap is not None:
            cap.release()

class DemoWindow(QMainWindow):
    def __init__(self, config_path):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        disp = self.config.get("display_config", {})
        self.setWindowTitle(disp.get("display_label", "YOLO Multi Demo"))
        w = disp.get("output_width", 1920)
        h = disp.get("output_height", 1080)
        self.resize(w, h)
        
        self.sources = self.config.get("video_sources", [])
        num_sources = len(self.sources)
        grid_cols = math.ceil(math.sqrt(num_sources))
        grid_rows = math.ceil(num_sources / grid_cols)
        
        main_widget = QWidget()
        layout = QGridLayout(main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        self.setCentralWidget(main_widget)
        
        self.labels = []
        for i in range(num_sources):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: black; border: 1px solid #333;")
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            layout.addWidget(lbl, i // grid_cols, i % grid_cols)
            self.labels.append(lbl)
            
        self.fps_label = QLabel(self)
        self.fps_label.move(10, 10)
        self.fps_label.setStyleSheet("background-color: black; color: white; padding: 5px; font-size: 16px;")
        self.fps_label.setText("FPS: 0.0")
        
        self.running = [True]
        self.workers = []
        model_path = self.config.get("model_path", "")
        
        target_w = w // grid_cols
        target_h = h // grid_rows
        
        for i, src_info in enumerate(self.sources):
            src_path = src_info[0]
            src_type = src_info[1]
            num_frames = src_info[2] if len(src_info) > 2 else -1
            worker = MultiWorker(i, model_path, src_path, src_type, num_frames, self.running, target_w, target_h)
            self.workers.append(worker)
            worker.start()
            
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(33) # ~30 fps
        
    def update_frames(self):
        for i, worker in enumerate(self.workers):
            with worker.lock:
                frame = worker.latest_frame
            if frame is not None:
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
                pix = QPixmap.fromImage(qimg)
                self.labels[i].setPixmap(pix)
                
        total_fps = sum(w.avg_fps for w in self.workers)
        num_w = max(1, len(self.workers))
        self.fps_label.setText(f"        AI Model : YOLOv5         FPS : {total_fps:.2f}      FPS/Stream: {total_fps/num_w:.2f}   ")
        self.fps_label.adjustSize()

    def closeEvent(self, event):
        self.running[0] = False
        for w in self.workers:
            w.join()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q or event.key() == Qt.Key.Key_Escape:
            self.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--exit-btn", action="store_true")
    args = parser.parse_args()
    
    # Notify launcher
    path = os.environ.get("DX_LAUNCHER_READY_FILE")
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                f.write("ready\n")
        except OSError:
            pass
            
    app = QApplication(sys.argv)
    win = DemoWindow(args.config)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
