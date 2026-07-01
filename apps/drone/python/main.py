import argparse
import math
import sys
import time
import cv2
import numpy as np
try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    from dx_engine import InferenceEngine
except ImportError:
    InferenceEngine = None
from PyQt5.QtCore import Qt, QTimer, QPoint, QRectF, QPointF
from PyQt5.QtGui import QImage, QPainter, QColor, QPen, QFont, QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QApplication, QWidget

# Constants
TEMPLATE_SIZE = 112
SEARCH_SIZE = 224
TEMPLATE_FACTOR = 2.0
SEARCH_FACTOR = 4.5
DEFAULT_UPDATE_INTERVAL = 200
TEMPLATE_UPDATE_THRESHOLD = 0.5
MAX_SCORE_DECAY = 1.0
CLIP_MARGIN = 10.0
EXIT_BTN_WIDTH = 32
EXIT_BTN_HEIGHT = 28
EXIT_BTN_MARGIN = 14

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def mat_to_qimage(bgr_img):
    if bgr_img is None or bgr_img.size == 0:
        return QImage()
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

class MixFormerV2Tracker:
    def __init__(self, model_path, backend_name="onnx"):
        self.state = (0, 0, 0, 0) # x, y, w, h
        self.frame_id = 0
        self.max_pred_score = -1.0
        self.backend_name = backend_name
        self.is_dxnn = (backend_name == "dxnn")

        if self.is_dxnn:
            if InferenceEngine is None:
                raise RuntimeError("dx_engine module not found. Cannot run dxnn backend.")
            print(f"[Info] Loading DXNN model for DEEPX NPU from: {model_path}")
            self.session = InferenceEngine(model_path)
            self.input_names = self.session.get_input_tensor_names()
            print(f"[Info] DXNN input order: {' '.join(self.input_names)}")
        else:
            if ort is None:
                raise RuntimeError("onnxruntime module not found. Cannot run onnx backend.")
            print(f"[Info] Loading ONNX model from: {model_path}")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, options, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_names = [o.name for o in self.session.get_outputs()]
        
        if len(self.input_names) < 3:
            raise RuntimeError("Model must have template, online_template, and search inputs")

    def init(self, image, init_bbox):
        self.state = init_bbox
        self.frame_id = 0
        self.max_pred_score = -1.0
        
        self.template_tensor, _ = self.sample_target(image, self.state, TEMPLATE_FACTOR, TEMPLATE_SIZE)
        self.online_template_tensor = self.template_tensor.copy()
        self.online_max_template_tensor = self.template_tensor.copy()

    def update(self, image):
        self.frame_id += 1
        
        search_tensor, resize_factor = self.sample_target(image, self.state, SEARCH_FACTOR, SEARCH_SIZE)
        inputs = self.prepare_inputs(search_tensor)
        
        if self.is_dxnn:
            job_id = self.session.run_async([inputs])
            outputs = self.session.wait(job_id)
        else:
            outputs = self.session.run(self.output_names, inputs)
        
        pred_box = self.find_pred_box(outputs)
        if pred_box is None:
            raise RuntimeError("Could not find Bounding Box output with last dimension 4")
            
        pred_score = self.find_pred_score(outputs)
        
        pred_box = pred_box * SEARCH_SIZE / resize_factor
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), image.shape[0], image.shape[1], CLIP_MARGIN)
        
        self.update_online_template(image, pred_score)
        
        return (int(self.state[0]), int(self.state[1]), int(self.state[2]), int(self.state[3]))

    def sample_target(self, image, target_box, search_area_factor, output_size):
        x, y, w, h = target_box
        if w <= 0.0 or h <= 0.0:
            raise RuntimeError("too small bounding box")

        crop_size = int(math.ceil(math.sqrt(w * h) * search_area_factor))
        if crop_size < 1:
            raise RuntimeError("too small bounding box")
        
        x1 = int(math.floor(x + 0.5 * w - 0.5 * crop_size + 0.5))
        y1 = int(math.floor(y + 0.5 * h - 0.5 * crop_size + 0.5))
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - image.shape[1] + 1)
        pad_bottom = max(0, y2 - image.shape[0] + 1)

        crop_x1 = x1 + pad_left
        crop_y1 = y1 + pad_top
        crop_x2 = x2 - pad_right
        crop_y2 = y2 - pad_bottom

        roi_x = int(np.clip(crop_x1, 0, image.shape[1]))
        roi_y = int(np.clip(crop_y1, 0, image.shape[0]))
        roi_right = int(np.clip(crop_x2, 0, image.shape[1]))
        roi_bottom = int(np.clip(crop_y2, 0, image.shape[0]))

        if roi_right - roi_x <= 0 or roi_bottom - roi_y <= 0:
            raise RuntimeError("empty crop")

        cropped = image[roi_y:roi_bottom, roi_x:roi_right]
        padded = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        resized = cv2.resize(padded, (output_size, output_size))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Build CHW flat tensor exactly like C++ does:
        # tensor[c * plane_size + y * output_size + x] = row[x][c]
        plane_size = output_size * output_size
        tensor = np.empty(3 * plane_size, dtype=np.float32)
        for c in range(3):
            tensor[c * plane_size : (c + 1) * plane_size] = resized[:, :, c].flatten()
        
        return tensor, float(output_size) / crop_size

    def map_box_back(self, pred_box, resize_factor):
        x, y, w, h = self.state
        cx_prev = x + 0.5 * w
        cy_prev = y + 0.5 * h
        half_side = 0.5 * SEARCH_SIZE / resize_factor
        
        cx_real = pred_box[0] + (cx_prev - half_side)
        cy_real = pred_box[1] + (cy_prev - half_side)
        
        return (cx_real - 0.5 * pred_box[2], cy_real - 0.5 * pred_box[3], pred_box[2], pred_box[3])

    def clip_box(self, box, image_h, image_w, margin):
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        x1 = np.clip(x1, 0.0, image_w - margin)
        x2 = np.clip(x2, margin, image_w)
        y1 = np.clip(y1, 0.0, image_h - margin)
        y2 = np.clip(y2, margin, image_h)

        return (x1, y1, max(margin, x2 - x1), max(margin, y2 - y1))

    def update_online_template(self, image, pred_score):
        if pred_score >= 0.0:
            self.max_pred_score *= MAX_SCORE_DECAY
            if pred_score > TEMPLATE_UPDATE_THRESHOLD and pred_score > self.max_pred_score:
                self.online_max_template_tensor, _ = self.sample_target(image, self.state, TEMPLATE_FACTOR, TEMPLATE_SIZE)
                self.max_pred_score = pred_score

        if self.frame_id > 0 and self.frame_id % DEFAULT_UPDATE_INTERVAL == 0:
            self.online_template_tensor = self.online_max_template_tensor.copy()
            self.max_pred_score = -1.0
            self.online_max_template_tensor = self.template_tensor.copy()

    def prepare_inputs(self, search_tensor):
        if self.is_dxnn:
            # dx_engine Python binding expects a single contiguous flat buffer
            # where all inputs are concatenated in the expected physical order:
            # [template, online_template, search]
            concat_tensor = np.concatenate([
                self.template_tensor,
                self.online_template_tensor,
                search_tensor
            ])
            return np.ascontiguousarray(concat_tensor, dtype=np.float32)

        inputs = {}
        for i, name in enumerate(self.input_names):
            lower = name.lower()
            if "online" in lower:
                data = self.online_template_tensor
                shape = (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)
            elif "search" in lower:
                data = search_tensor
                shape = (1, 3, SEARCH_SIZE, SEARCH_SIZE)
            elif "template" in lower:
                data = self.template_tensor
                shape = (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)
            else:
                if i == 0:
                    data = self.template_tensor
                    shape = (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)
                elif i == 1:
                    data = self.online_template_tensor
                    shape = (1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE)
                else:
                    data = search_tensor
                    shape = (1, 3, SEARCH_SIZE, SEARCH_SIZE)

            # ONNX runtime expects 4D (N, C, H, W) shaped tensor
            inputs[name] = data.reshape(shape).astype(np.float32)
                
        return inputs

    def find_pred_box(self, outputs):
        for out in outputs:
            if out.shape[-1] == 4 or out.size == 4:
                return out.reshape(-1, 4).mean(axis=0)
        return None

    def find_pred_score(self, outputs):
        for out in outputs:
            if out.shape[-1] != 4 or out.size == 1:
                if out.size > 0:
                    return sigmoid(out.flatten()[0])
        return -1.0

class TrackingWindow(QWidget):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle(f"MixFormerV2 Tracking ({args.backend.upper()} - Python)")
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(640, 360)
        
        self.cap = cv2.VideoCapture(args.video)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {args.video}")
            
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        self.frame_interval_ms = max(1, int(round(1000.0 / fps)))
        
        ret, self.current_frame = self.cap.read()
        if not ret or self.current_frame is None:
            raise RuntimeError(f"cannot read first frame from video: {args.video}")
            
        self.frame_image = mat_to_qimage(self.current_frame)
        self.tracker = None
        if args.backend == "onnx" and not args.model.endswith('.onnx'):
            print(f"[Warning] Provided model '{args.model}' is not an ONNX file. Proceeding without tracking to show UI.")
        elif args.backend == "dxnn" and not args.model.endswith('.dxnn'):
            print(f"[Warning] Provided model '{args.model}' is not a DXNN file. Proceeding without tracking to show UI.")
        else:
            self.tracker = MixFormerV2Tracker(args.model, args.backend)
            
        self.mode = "Selecting"
        self.dragging = False
        self.drag_start = None
        self.drag_current = None
        self.selected_roi = None
        self.latest_bbox = None
        self.display_fps = 0.0
        self.last_frame_time = time.time()
        self.status_text = ""
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_frame)
        self.timer.setTimerType(Qt.PreciseTimer)
        
        print(f"[{args.backend.upper()} Mode] Drag to select the object to track; tracking starts on release.")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 10, 12))
        
        if self.frame_image.isNull():
            return
            
        image_rect = self.image_draw_rect()
        painter.drawImage(image_rect, self.frame_image)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        self.draw_tracking_overlay(painter, image_rect)
        self.draw_selection_overlay(painter, image_rect)
        self.draw_hud(painter, image_rect)
        self.draw_exit_button(painter)

    def mousePressEvent(self, event):
        if self.args.exit_btn and event.button() == Qt.LeftButton and self.exit_button_rect().contains(event.pos()):
            QApplication.quit()
            return
            
        if self.mode != "Selecting" or event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
            
        img_pt = self.widget_to_image(event.pos())
        if img_pt is None:
            return
            
        self.dragging = True
        self.drag_start = img_pt
        self.drag_current = img_pt
        self.selected_roi = None
        self.update()

    def mouseMoveEvent(self, event):
        if self.mode == "Selecting" and self.dragging:
            self.drag_current = self.widget_to_image_clamped(event.pos())
            self.selected_roi = self.normalized_rect(self.drag_start, self.drag_current)
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "Selecting" and self.dragging and event.button() == Qt.LeftButton:
            self.drag_current = self.widget_to_image_clamped(event.pos())
            self.selected_roi = self.normalized_rect(self.drag_start, self.drag_current)
            self.dragging = False
            self.start_tracking()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Escape, Qt.Key_Q):
            QApplication.quit()
            return
        if event.key() == Qt.Key_F:
            if self.isFullScreen(): self.showNormal()
            else: self.showFullScreen()
            return
        super().keyPressEvent(event)

    def exit_button_rect(self):
        x = max(0, self.width() - EXIT_BTN_WIDTH - EXIT_BTN_MARGIN)
        return QRectF(x, EXIT_BTN_MARGIN, EXIT_BTN_WIDTH, EXIT_BTN_HEIGHT)

    def draw_exit_button(self, painter):
        if not self.args.exit_btn: return
        rect = self.exit_button_rect()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        painter.setBrush(QColor(48, 45, 45, 230))
        painter.drawRoundedRect(rect, 6, 6)
        
        font = painter.font()
        font.setPixelSize(13)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(204, 204, 204))
        painter.drawText(rect, Qt.AlignCenter, "X")
        painter.restore()

    def image_draw_rect(self):
        target_size = self.frame_image.size()
        target_size.scale(self.size(), Qt.KeepAspectRatio)
        tl = QPointF((self.width() - target_size.width()) * 0.5, (self.height() - target_size.height()) * 0.5)
        from PyQt5.QtCore import QSizeF
        return QRectF(tl, QSizeF(target_size))

    def widget_to_image(self, widget_point):
        rect = self.image_draw_rect()
        if not rect.contains(QPointF(widget_point)):
            return None
        return self.widget_to_image_clamped(widget_point)

    def widget_to_image_clamped(self, widget_point):
        rect = self.image_draw_rect()
        x = np.clip(widget_point.x(), rect.left(), rect.right())
        y = np.clip(widget_point.y(), rect.top(), rect.bottom())
        sx = self.frame_image.width() / rect.width()
        sy = self.frame_image.height() / rect.height()
        return (x - rect.left()) * sx, (y - rect.top()) * sy

    def image_to_widget_rect(self, img_rect, draw_rect):
        x, y, w, h = img_rect
        sx = draw_rect.width() / self.frame_image.width()
        sy = draw_rect.height() / self.frame_image.height()
        return QRectF(draw_rect.left() + x * sx, draw_rect.top() + y * sy, w * sx, h * sy)

    def normalized_rect(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        return (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

    def draw_tracking_overlay(self, painter, draw_rect):
        if self.mode not in ("Tracking", "Finished") or not self.latest_bbox:
            return
        if self.latest_bbox[2] <= 0 or self.latest_bbox[3] <= 0:
            return
        painter.setPen(QPen(QColor(38, 230, 118), 3))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.image_to_widget_rect(self.latest_bbox, draw_rect))

    def draw_selection_overlay(self, painter, draw_rect):
        if self.mode != "Selecting": return
        roi = self.selected_roi
        if self.dragging:
            roi = self.normalized_rect(self.drag_start, self.drag_current)
        if not roi or roi[2] <= 0 or roi[3] <= 0:
            return
        painter.setPen(QPen(QColor(255, 214, 88), 2))
        painter.setBrush(QColor(255, 214, 88, 42))
        painter.drawRect(self.image_to_widget_rect(roi, draw_rect))

    def draw_hud(self, painter, draw_rect):
        panel = QRectF(draw_rect.left() + 16.0, draw_rect.top() + 14.0, 360.0, 72.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(panel, 8, 8)
        
        font = painter.font()
        font.setPixelSize(20)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor(235, 250, 241))
        painter.drawText(panel.adjusted(14, 10, -12, -36), Qt.AlignLeft | Qt.AlignVCenter, f"MixFormerV2 ({self.args.backend.upper()})")
        
        font.setPixelSize(14)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(188, 205, 196))
        
        if self.mode == "Selecting": status = "Drag over a target to start tracking"
        elif self.mode == "Tracking": status = f"Tracking  |  {self.display_fps:.1f} FPS"
        elif self.mode == "Finished": status = "Finished"
        else: status = self.status_text
        
        painter.drawText(panel.adjusted(14, 34, -12, -10), Qt.AlignLeft | Qt.AlignVCenter, status)

    def start_tracking(self):
        if not self.selected_roi or self.selected_roi[2] < 2.0 or self.selected_roi[3] < 2.0:
            print("Invalid bounding box. Select a larger target region.")
            self.update()
            return
            
        if self.tracker:
            self.tracker.init(self.current_frame, self.selected_roi)
        self.latest_bbox = self.selected_roi
        self.mode = "Tracking"
        self.last_frame_time = time.time()
        self.timer.start(self.frame_interval_ms)
        self.update()

    def process_next_frame(self):
        if self.mode != "Tracking": return
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if self.args.loop:
                    self.restart_tracking_loop()
                    return
                self.timer.stop()
                self.mode = "Finished"
                self.update()
                return
                
            if self.tracker:
                self.latest_bbox = self.tracker.update(frame)
            self.current_frame = frame
            
            # Fast cv2 resizing for rendering
            lbl_w, lbl_h = self.width(), self.height()
            if lbl_w > 0 and lbl_h > 0:
                aspect_ratio = self.current_frame.shape[1] / self.current_frame.shape[0]
                if lbl_w / lbl_h > aspect_ratio:
                    new_h = lbl_h
                    new_w = int(new_h * aspect_ratio)
                else:
                    new_w = lbl_w
                    new_h = int(new_w / aspect_ratio)
                resized_frame = cv2.resize(self.current_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                resized_frame = self.current_frame
                
            self.frame_image = mat_to_qimage(resized_frame)
            
            self.update_fps()
            self.update()
        except Exception as e:
            self.timer.stop()
            self.mode = "Error"
            self.status_text = str(e)
            print(f"Error: {e}")
            self.update()

    def restart_tracking_loop(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.release()
            self.cap = cv2.VideoCapture(self.args.video)
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise RuntimeError("cannot restart video loop")
                
        self.current_frame = frame
        self.frame_image = mat_to_qimage(self.current_frame)
        if self.tracker:
            self.tracker.init(self.current_frame, self.selected_roi)
        self.latest_bbox = self.selected_roi
        self.display_fps = 0.0
        self.last_frame_time = time.time()
        self.update()

    def update_fps(self):
        now = time.time()
        seconds = now - self.last_frame_time
        self.last_frame_time = now
        if seconds > 0.0:
            instant = 1.0 / seconds
            if self.display_fps <= 0.0:
                self.display_fps = instant
            else:
                self.display_fps = self.display_fps * 0.88 + instant * 0.12

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="dxnn")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--full_screen", action="store_true")
    parser.add_argument("--exit-btn", action="store_true", dest="exit_btn")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = TrackingWindow(args)
    if args.full_screen:
        window.showFullScreen()
    else:
        window.resize(1280, 720)
        window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
