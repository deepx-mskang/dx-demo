import argparse
import sys
import time
import cv2
import numpy as np
import os

# Try importing dx_engine, fallback to onnxruntime if not available or requested
try:
    import dx_engine
    DX_AVAILABLE = True
except ImportError:
    DX_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
import threading
import time
import sys
import argparse
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils/tensor_harness'))
try:
    import tensor_harness
except ImportError:
    tensor_harness = None

class DepthEstimationApp:
    def __init__(self, model_path, backend):
        self.backend = backend.lower()
        self.is_dxnn = self.backend == 'dxnn'
        
        if self.is_dxnn and not DX_AVAILABLE:
            raise RuntimeError("DXNN backend requested but dx_engine is not installed.")
        if not self.is_dxnn and not ORT_AVAILABLE:
            raise RuntimeError("ONNX backend requested but onnxruntime is not installed.")
            
        if self.is_dxnn:
            print(f"[Info] Loading DXNN model for DEEPX NPU from: {model_path}")
            self.session = dx_engine.InferenceEngine(model_path)
            
            # Use info dict
            input_info = self.session.get_input_tensors_info()
            if not input_info:
                # Fallback shapes if API fails
                self.net_w = 518
                self.net_h = 518
                self.nchw = True
            else:
                shape = input_info[0]['shape']
                if len(shape) == 4:
                    if shape[3] == 3:
                        self.nchw = False
                        self.net_h = shape[1]
                        self.net_w = shape[2]
                    else:
                        self.nchw = True
                        self.net_h = shape[2]
                        self.net_w = shape[3]
                else:
                    self.net_w = 518
                    self.net_h = 518
                    self.nchw = True

        else:
            print(f"[Info] Loading ONNX model using ONNXRuntime from: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4:
                if shape[3] == 3:
                    self.nchw = False
                    self.net_h = shape[1]
                    self.net_w = shape[2]
                else:
                    self.nchw = True
                    self.net_h = shape[2]
                    self.net_w = shape[3]
            else:
                self.net_w = 518
                self.net_h = 518
                self.nchw = True

        print(f"Model Input Size: {self.net_w}x{self.net_h}")

    def preprocess(self, frame_bgr):
        # 1. Convert BGR to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Resize
        resized = cv2.resize(rgb, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        
        # 3. Cast to float32 and scale to [0, 1]
        f32 = resized.astype(np.float32) / 255.0
        
        # 4. Normalize
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        stddev = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        normalized = (f32 - mean) / stddev
        
        # 5. Format to NCHW or NHWC
        if self.nchw:
            # HWC to CHW
            normalized = np.transpose(normalized, (2, 0, 1))
            
        # Ensure memory contiguity as defined by the Harness Rules
        # This is critical for DXNN correctness!
        tensor = np.ascontiguousarray(normalized.flatten(), dtype=np.float32)
            
        return tensor

    def postprocess(self, output_tensor):
        # Flatten and reshape if necessary
        depth = output_tensor.squeeze()
        
        min_v = np.min(depth)
        max_v = np.max(depth)
        
        if max_v - min_v > 0.0:
            normalized = (depth - min_v) / (max_v - min_v) * 255.0
            normalized = normalized.astype(np.uint8)
        else:
            normalized = np.zeros_like(depth, dtype=np.uint8)
            
        bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        return bgr

    def submit(self, image):
        # Preprocessing
        input_tensor = self.preprocess(image)
        # Inference
        if self.is_dxnn:
            job_id = self.session.run_async([input_tensor])
            return job_id, image
        else:
            # ONNX expects 4D shape
            shape = (1, 3, self.net_h, self.net_w) if self.nchw else (1, self.net_h, self.net_w, 3)
            tensor_4d = input_tensor.reshape(shape)
            outputs = self.session.run(self.output_names, {self.input_names[0]: tensor_4d})
            return outputs, image

    def wait_and_postprocess(self, submit_info):
        if self.is_dxnn:
            job_id, image = submit_info
            outputs = self.session.wait(job_id)
            output_data = outputs[0]
        else:
            outputs, image = submit_info
            output_data = outputs[0]

        # Postprocessing
        depth_bgr = self.postprocess(output_data)
        return depth_bgr, image

import queue

class InferWorker:
    def __init__(self, app_model, cap, target_w, target_h):
        self.app_model = app_model
        self.cap = cap
        self.target_w = target_w
        self.target_h = target_h
        self.running = True
        self.latest_frame = None
        self.lock = threading.Lock()
        self.avg_fps = 0.0

        self.submit_queue = queue.Queue(maxsize=15)
        self.submit_thread = threading.Thread(target=self._submit_loop)
        self.wait_thread = threading.Thread(target=self._wait_loop)

    def start(self):
        self.submit_thread.start()
        self.wait_thread.start()

    def _submit_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            submit_info = self.app_model.submit(frame)
            start_time = time.time()
            self.submit_queue.put((submit_info, start_time))

    def _wait_loop(self):
        fps_values = []
        last_time = time.time()
        while self.running:
            try:
                submit_info, start_time = self.submit_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            depth_bgr, frame = self.app_model.wait_and_postprocess(submit_info)
            
            end_time = time.time()
            fps = 1.0 / (end_time - last_time + 1e-6)
            last_time = end_time
            
            fps_values.append(fps)
            if len(fps_values) > 30:
                fps_values.pop(0)
            avg_fps = sum(fps_values) / len(fps_values)
            self.avg_fps = avg_fps
            
            depth_bgr_resized = cv2.resize(depth_bgr, (frame.shape[1], frame.shape[0]))
            combined = np.hstack((frame, depth_bgr_resized))
            
            cv2.putText(combined, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.target_w > 0 and self.target_h > 0:
                combined = cv2.resize(combined, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            
            rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            
            with self.lock:
                self.latest_frame = rgb

    def stop(self):
        self.running = False
        self.submit_thread.join()
        self.wait_thread.join()

class DepthDemoWindow(QMainWindow):
    def __init__(self, app_model, cap, full_screen):
        super().__init__()
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        
        self.target_w, self.target_h = 1280, 720
        if full_screen:
            self.showFullScreen()
            screen = QApplication.primaryScreen().geometry()
            self.target_w, self.target_h = screen.width(), screen.height()
        else:
            self.resize(1280, 720)
            self.show()
            
        self.worker = InferWorker(app_model, cap, self.target_w, self.target_h)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        with self.worker.lock:
            rgb = self.worker.latest_frame
        
        if rgb is not None:
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(qimg))

    def closeEvent(self, event):
        self.worker.stop()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape or event.key() == Qt.Key_Q:
            self.close()

def main():
    parser = argparse.ArgumentParser(description="Depth Estimation Demo")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.onnx or .dxnn)")
    parser.add_argument("--video", type=str, default="", help="Path to the input video file")
    parser.add_argument("--camera", type=int, default=-1, help="Camera device ID (e.g., 0 or 1)")
    parser.add_argument("--backend", type=str, default="dxnn", choices=["dxnn", "onnx"], help="Backend to use")
    parser.add_argument("--full_screen", action="store_true", help="Run in fullscreen mode")
    parser.add_argument("--exit-btn", action="store_true", help="Show exit button overlay")
    args = parser.parse_args()

    app_model = DepthEstimationApp(args.model, args.backend)
    
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        print("Error: Must provide either --video or --camera")
        sys.exit(1)
        
    if not cap.isOpened():
        print(f"Error: Could not open video/camera")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = DepthDemoWindow(app_model, cap, args.full_screen)
    app.exec_()
    cap.release()

if __name__ == "__main__":
    main()
