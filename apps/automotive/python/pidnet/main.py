import argparse
import sys
import time
import cv2
import numpy as np
import os

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
import queue
import time
import sys
import argparse
import numpy as np
import cv2

class PIDNetApp:
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
            
            input_info = self.session.get_input_tensors_info()
            if not input_info:
                self.net_w, self.net_h = 2048, 1024
                self.nchw = False
                self.dtype = np.uint8
            else:
                shape = input_info[0]['shape']
                # e.g., shape=[1, 1024, 2048, 3] -> NHWC
                if len(shape) == 4:
                    if shape[3] == 3:
                        self.nchw = False
                        self.net_h, self.net_w = shape[1], shape[2]
                    else:
                        self.nchw = True
                        self.net_h, self.net_w = shape[2], shape[3]
                else:
                    self.net_w, self.net_h = 2048, 1024
                    self.nchw = False
                
                # Check data type
                self.dtype = input_info[0]['dtype']

        else:
            print(f"[Info] Loading ONNX model using ONNXRuntime from: {model_path}")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.output_names = [out.name for out in self.session.get_outputs()]
            
            shape = self.session.get_inputs()[0].shape
            if len(shape) == 4:
                if shape[3] == 3:
                    self.nchw = False
                    self.net_h, self.net_w = shape[1], shape[2]
                else:
                    self.nchw = True
                    self.net_h, self.net_w = shape[2], shape[3]
            else:
                self.net_w, self.net_h = 2048, 1024
                self.nchw = False
            self.dtype = np.uint8 # Fallback

        print(f"Model Input Size: {self.net_w}x{self.net_h}, NCHW: {self.nchw}, Dtype: {self.dtype}")

        # Setup palette
        self.palette = self.get_palette("pastel")

    def get_palette(self, name="pastel"):
        # Dummy palette for 19 classes
        np.random.seed(42)
        return np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)

    def preprocess(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        
        if self.dtype == np.float32:
            resized = resized.astype(np.float32) / 255.0
            
        if self.nchw:
            resized = np.transpose(resized, (2, 0, 1))
            
        tensor = np.ascontiguousarray(resized.flatten(), dtype=self.dtype)
        return tensor

    def postprocess(self, output_tensor, orig_w, orig_h):
        # output is likely a class map or logits
        # For PIDNet, shape is typically [1, 19, H, W] or [1, H, W, 19] (logits), or [1, H, W] (class map)
        
        if self.is_dxnn:
            out_info = self.session.get_output_tensors_info()
            out_shape = out_info[0]['shape']
        else:
            out_shape = self.session.get_outputs()[0].shape
            
        # Parse shape
        if len(out_shape) == 4:
            if out_shape[1] <= out_shape[3]:
                # NCHW
                logits = output_tensor.reshape(out_shape[1], out_shape[2], out_shape[3])
                class_map = np.argmax(logits, axis=0) # [H, W]
            else:
                # NHWC
                logits = output_tensor.reshape(out_shape[1], out_shape[2], out_shape[3])
                class_map = np.argmax(logits, axis=2) # [H, W]
        elif len(out_shape) == 3:
            class_map = output_tensor.reshape(out_shape[1], out_shape[2])
        elif len(out_shape) == 2:
            class_map = output_tensor.reshape(out_shape[0], out_shape[1])
        else:
            class_map = output_tensor
            
        class_map = class_map.astype(np.uint8)
        
        # Resize to original
        resized_mask = cv2.resize(class_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # Apply palette
        color_mask = self.palette[resized_mask]
        return color_mask

    def submit(self, image):
        orig_h, orig_w = image.shape[:2]
        input_tensor = self.preprocess(image)

        if self.is_dxnn:
            job_id = self.session.run_async([input_tensor])
            return job_id, image, orig_w, orig_h
        else:
            shape = (1, 3, self.net_h, self.net_w) if self.nchw else (1, self.net_h, self.net_w, 3)
            tensor_4d = input_tensor.reshape(shape)
            outputs = self.session.run(self.output_names, {self.input_names[0]: tensor_4d})
            return outputs, image, orig_w, orig_h

    def wait_and_postprocess(self, submit_info):
        if self.is_dxnn:
            job_id, image, orig_w, orig_h = submit_info
            outputs = self.session.wait(job_id)
            output_data = outputs[0]
        else:
            outputs, image, orig_w, orig_h = submit_info
            output_data = outputs[0]

        color_mask = self.postprocess(output_data, orig_w, orig_h)
        blended = cv2.addWeighted(image, 0.5, color_mask, 0.5, 0)
        return blended

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
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0.0
        while self.running:
            try:
                submit_info, start_time = self.submit_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            blended = self.app_model.wait_and_postprocess(submit_info)
            
            self.frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.avg_fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_start_time = time.time()
            
            cv2.putText(blended, f"FPS: {self.avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if self.target_w > 0 and self.target_h > 0:
                blended = cv2.resize(blended, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
                
            rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
            
            with self.lock:
                self.latest_frame = rgb

    def stop(self):
        self.running = False
        self.submit_thread.join()
        self.wait_thread.join()

class DemoWindow(QMainWindow):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="dxnn")
    parser.add_argument("--full_screen", action="store_true")
    parser.add_argument("--exit-btn", action="store_true")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--seg-palette", type=str, default="pastel")
    args = parser.parse_args()

    app_model = PIDNetApp(args.model, args.backend)
    
    if args.camera >= 0:
        cap = cv2.VideoCapture(args.camera)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        sys.exit(1)
        
    app = QApplication(sys.argv)
    window = DemoWindow(app_model, cap, args.full_screen)
    app.exec_()
    cap.release()

if __name__ == "__main__":
    main()
