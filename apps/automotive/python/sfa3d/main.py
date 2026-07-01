import argparse
import sys
import time
import os
import glob
import math

import numpy as np
import cv2

try:
    import dx_engine
    DX_AVAILABLE = True
except ImportError:
    DX_AVAILABLE = False

import threading
import queue
import concurrent.futures

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget
# SFA3D Constants
BEV_WIDTH = 608
BEV_HEIGHT = 608
DOWN_RATIO = 4

kFrontBoundary = {'minX': 0.0, 'maxX': 50.0, 'minY': -25.0, 'maxY': 25.0, 'minZ': -2.73, 'maxZ': 1.27}
kBackBoundary = {'minX': -50.0, 'maxX': 0.0, 'minY': -25.0, 'maxY': 25.0, 'minZ': -2.73, 'maxZ': 1.27}

class Calibration:
    def __init__(self, calib_path):
        self.p2 = np.zeros(12)
        self.r0 = np.zeros(9)
        self.v2c = np.zeros(12)
        with open(calib_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                key = parts[0]
                if key.endswith(':'):
                    key = key[:-1]
                values = np.array([float(x) for x in parts[1:]])
                if key == 'P2':
                    self.p2 = values
                elif key == 'R0_rect':
                    self.r0 = values
                elif key == 'Tr_velo_to_cam':
                    self.v2c = values

def lidar_to_camera(x, y, z, calib):
    ref_x = calib.v2c[0]*x + calib.v2c[1]*y + calib.v2c[2]*z + calib.v2c[3]
    ref_y = calib.v2c[4]*x + calib.v2c[5]*y + calib.v2c[6]*z + calib.v2c[7]
    ref_z = calib.v2c[8]*x + calib.v2c[9]*y + calib.v2c[10]*z + calib.v2c[11]

    cx = calib.r0[0]*ref_x + calib.r0[1]*ref_y + calib.r0[2]*ref_z
    cy = calib.r0[3]*ref_x + calib.r0[4]*ref_y + calib.r0[5]*ref_z
    cz = calib.r0[6]*ref_x + calib.r0[7]*ref_y + calib.r0[8]*ref_z
    return cx, cy, cz

def compute_box_3d(det):
    c = math.cos(det['yaw'])
    s = math.sin(det['yaw'])
    l = det['l']
    w = det['w']
    h = det['h']
    
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = []
    for i in range(8):
        cx = c * x_corners[i] + s * z_corners[i] + det['x']
        cy = y_corners[i] + det['y']
        cz = -s * x_corners[i] + c * z_corners[i] + det['z']
        corners.append((cx, cy, cz))
    return corners

def project_to_image(corners, calib):
    projected = []
    for pt in corners:
        u = calib.p2[0]*pt[0] + calib.p2[1]*pt[1] + calib.p2[2]*pt[2] + calib.p2[3]
        v = calib.p2[4]*pt[0] + calib.p2[5]*pt[1] + calib.p2[6]*pt[2] + calib.p2[7]
        w_p = calib.p2[8]*pt[0] + calib.p2[9]*pt[1] + calib.p2[10]*pt[2] + calib.p2[11]
        
        # To avoid division by zero or negative depths being projected improperly
        if w_p > 0:
            projected.append((int(u / w_p), int(v / w_p)))
        else:
            projected.append((-1, -1))
    return projected

def draw_box_3d(image, projected, color):
    # Verify all corners are valid
    for pt in projected:
        if pt[0] == -1:
            return
            
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for e in edges:
        cv2.line(image, projected[e[0]], projected[e[1]], color, 2)
    # Draw heading
    cv2.line(image, projected[0], projected[5], color, 1)
    cv2.line(image, projected[1], projected[4], color, 1)

def show_rgb_image_with_boxes(img, detections, calib, boundary):
    colors = [(64, 230, 255), (118, 238, 92), (224, 70, 255)] # Aurora palette (BGR)
    bound_size_x = boundary['maxX'] - boundary['minX']
    bound_size_y = boundary['maxY'] - boundary['minY']
    min_x = boundary['minX']
    min_y = boundary['minY']
    min_z = boundary['minZ']
    
    for cls in range(3):
        for raw_det in detections[cls]:
            # Convert BEV coords to Lidar real coordinates
            lidar_x = raw_det['y'] / float(BEV_HEIGHT) * bound_size_x + min_x
            lidar_y = raw_det['x'] / float(BEV_WIDTH) * bound_size_y + min_y
            lidar_z = raw_det['z'] + min_z
            lidar_w = raw_det['w'] / float(BEV_WIDTH) * bound_size_y
            lidar_l = raw_det['l'] / float(BEV_HEIGHT) * bound_size_x
            lidar_h = raw_det['h']
            lidar_yaw = -raw_det['yaw']
            
            # Convert lidar to camera coords
            cx, cy, cz = lidar_to_camera(lidar_x, lidar_y, lidar_z, calib)
            
            if cz < 2.0:
                continue
                
            det = raw_det.copy()
            det['x'] = cx
            det['y'] = cy
            det['z'] = cz
            det['w'] = lidar_w
            det['l'] = lidar_l
            det['h'] = lidar_h
            det['yaw'] = lidar_yaw - math.pi / 2.0
            
            corners_3d = compute_box_3d(det)
            projected = project_to_image(corners_3d, calib)
            draw_box_3d(img, projected, colors[cls])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def read_lidar_file(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

def make_bev_map(points, boundary):
    min_x, max_x = boundary['minX'], boundary['maxX']
    min_y, max_y = boundary['minY'], boundary['maxY']
    min_z, max_z = boundary['minZ'], boundary['maxZ']
    
    mask = (points[:, 0] >= min_x) & (points[:, 0] <= max_x) & \
           (points[:, 1] >= min_y) & (points[:, 1] <= max_y) & \
           (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    pts = points[mask]
    
    discretization = (max_x - min_x) / BEV_HEIGHT
    
    rows = np.floor(pts[:, 0] / discretization).astype(np.int32)
    cols = np.floor(pts[:, 1] / discretization + BEV_WIDTH / 2.0).astype(np.int32)
    
    valid = (rows >= 0) & (rows < BEV_HEIGHT) & (cols >= 0) & (cols < BEV_WIDTH)
    pts = pts[valid]
    rows = rows[valid]
    cols = cols[valid]
    
    density = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.float32)
    height_map = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.float32)
    intensity_map = np.zeros((BEV_HEIGHT, BEV_WIDTH), dtype=np.float32)
    
    sort_idx = np.argsort(pts[:, 2])
    pts = pts[sort_idx]
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    
    height_map[rows, cols] = pts[:, 2] - min_z
    intensity_map[rows, cols] = pts[:, 3]
    
    linear_indices = rows * BEV_WIDTH + cols
    unique_indices, counts = np.unique(linear_indices, return_counts=True)
    ur = unique_indices // BEV_WIDTH
    uc = unique_indices % BEV_WIDTH
    
    density[ur, uc] = np.minimum(1.0, np.log(counts + 1.0) / np.log(64.0))
    height_map /= (max_z - min_z)
    
    bev = np.stack([intensity_map, height_map, density], axis=-1) # [H, W, 3]
    bev = np.clip(bev * 255.0, 0, 255).astype(np.uint8)
    return bev

def decode_outputs(outputs, top_k=50, peak_thresh=0.2):
    # Map DXNN outputs to their expected names
    # Assuming outputs is already a dict since we can map it in infer_side
    pass

    hm_cen = outputs['hm_cen'][0]
    cen_offset = outputs['cen_offset'][0]
    direction = outputs['direction'][0]
    z_coor = outputs['z_coor'][0, 0]
    dim = outputs['dim'][0]

    hm_cen = sigmoid(hm_cen)
    cen_offset = sigmoid(cen_offset)

    hm_max = np.zeros_like(hm_cen)
    for i in range(hm_cen.shape[0]):
        hm_max[i] = cv2.dilate(hm_cen[i], np.ones((3, 3)))
    
    keep = (hm_max == hm_cen)
    
    candidates = []
    for cls in range(3):
        cls_hm = hm_cen[cls]
        cls_keep = keep[cls]
        y, x = np.where(cls_keep & (cls_hm > peak_thresh))
        scores = cls_hm[y, x]
        for i in range(len(scores)):
            candidates.append((scores[i], cls, y[i], x[i]))
            
    candidates.sort(key=lambda x: x[0], reverse=True)
    candidates = candidates[:top_k]
    
    detections = {0: [], 1: [], 2: []}
    for score, cls, y, x in candidates:
        center_x = x + cen_offset[0, y, x]
        center_y = y + cen_offset[1, y, x]
        dir_sin = direction[0, y, x]
        dir_cos = direction[1, y, x]
        
        det = {
            'score': score,
            'x': center_x * DOWN_RATIO,
            'y': center_y * DOWN_RATIO,
            'z': z_coor[y, x],
            'h': dim[0, y, x],
            'w': dim[1, y, x] / 50.0 * BEV_WIDTH,  # kBoundSizeY
            'l': dim[2, y, x] / 50.0 * BEV_HEIGHT, # kBoundSizeX
            'yaw': math.atan2(dir_sin, dir_cos)
        }
        detections[cls].append(det)
        
    return detections

def draw_bev(img, detections):
    # Aurora palette
    colors = [(64, 230, 255), (118, 238, 92), (224, 70, 255)] # BGR
    for cls in range(3):
        for det in detections[cls]:
            x, y, w, l, yaw = det['x'], det['y'], det['w'], det['l'], det['yaw']
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            
            pts = [
                (int(x - w/2 * cos_y - l/2 * sin_y), int(y - w/2 * sin_y + l/2 * cos_y)),
                (int(x - w/2 * cos_y + l/2 * sin_y), int(y - w/2 * sin_y - l/2 * cos_y)),
                (int(x + w/2 * cos_y + l/2 * sin_y), int(y + w/2 * sin_y - l/2 * cos_y)),
                (int(x + w/2 * cos_y - l/2 * sin_y), int(y + w/2 * sin_y + l/2 * cos_y))
            ]
            for i in range(4):
                cv2.line(img, pts[i], pts[(i+1)%4], colors[cls], 2)
            cv2.line(img, pts[0], pts[3], (255, 234, 98), 2) # Heading

class SFA3DApp:
    def __init__(self, model_path, dataset_dir):
        if not DX_AVAILABLE:
            raise RuntimeError("DXNN backend requested but dx_engine is not installed.")
            
        print(f"[Info] Loading DXNN model from: {model_path}")
        self.session = dx_engine.InferenceEngine(model_path)
        
        self.dataset_dir = dataset_dir
        # Find all .bin files
        self.lidar_files = sorted(glob.glob(os.path.join(dataset_dir, 'velodyne_points', 'data', '*.bin')))
        if not self.lidar_files:
            # Try to resolve relative to workspace
            self.lidar_files = sorted(glob.glob(os.path.join(dataset_dir, '..', '..', 'dataset', 'kitti', 'demo', '2011_09_26_drive_0014_sync', 'velodyne_points', 'data', '*.bin')))
            if not self.lidar_files:
                raise RuntimeError(f"No LIDAR .bin files found in {dataset_dir}")
                
        self.img_files = sorted(glob.glob(os.path.join(os.path.dirname(self.lidar_files[0]), '..', '..', 'image_02', 'data', '*.png')))
        self.idx = 0
        
        calib_path = os.path.abspath(os.path.join(dataset_dir, '../../..', 'calib.txt'))
        if not os.path.exists(calib_path):
            calib_path = os.path.join(dataset_dir, 'calib.txt')
        self.calib = Calibration(calib_path)
        
        input_info = self.session.get_input_tensors_info()
        self.dtype = input_info[0]['dtype']
        
        self.stop_event = threading.Event()
        self.decode_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=2)
        
        self.job_id_counter = 0
        self.pending_jobs = {}
        self.jobs_lock = threading.Lock()
        
        # Register callback
        self.session.register_callback(self._inference_callback)
        
        self.worker_thread = threading.Thread(target=self._data_prep_loop)
        self.decode_thread = threading.Thread(target=self._decode_loop)
        
        self.worker_thread.start()
        self.decode_thread.start()

    def stop(self):
        self.stop_event.set()
        
        # Cleanup DXNN session callbacks
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
        job_id, side = user_arg
        
        # We must clone the outputs here because the pointer might be freed by DXNN
        cloned_outputs = [np.array(o).copy() for o in outputs]
        
        self.decode_queue.put((job_id, side, cloned_outputs))
        return 0

    def _data_prep_loop(self):
        while not self.stop_event.is_set():
            with self.jobs_lock:
                num_pending = len(self.pending_jobs)
                
            if num_pending > 2 or self.result_queue.full():
                time.sleep(0.001)
                continue
            
            start_time = time.time()
                
            if self.idx >= len(self.lidar_files):
                self.idx = 0
                
            lidar_path = self.lidar_files[self.idx]
            img_path = self.img_files[self.idx]
            self.idx += 1
            
            points = read_lidar_file(lidar_path)
            img = cv2.imread(img_path)
            
            # Make BEV
            front_bev = make_bev_map(points, kFrontBoundary)
            back_bev = np.flip(make_bev_map(points, kBackBoundary), axis=(0, 1))
            
            job_id = self.job_id_counter
            self.job_id_counter += 1
            
            with self.jobs_lock:
                self.pending_jobs[job_id] = {
                    'img': img,
                    'front_bev': front_bev,
                    'back_bev': back_bev,
                    'start_time': start_time,
                    'front_outputs': None,
                    'back_outputs': None
                }
            
            # Submit async
            front_tensor = np.ascontiguousarray(front_bev.flatten(), dtype=self.dtype)
            back_tensor = np.ascontiguousarray(back_bev.flatten(), dtype=self.dtype)
            
            self.session.run_async([front_tensor], user_arg=(job_id, 'front'))
            self.session.run_async([back_tensor], user_arg=(job_id, 'back'))

    def _decode_outputs_raw(self, outputs):
        out_info = self.session.get_output_tensors_info()
        reshaped = {}
        for i, info in enumerate(out_info):
            reshaped[info['name']] = outputs[i].reshape(info['shape'])
        return decode_outputs(reshaped)

    def _decode_loop(self):
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.avg_fps = 0.0
        while not self.stop_event.is_set():
            try:
                job_id, side, outputs = self.decode_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            with self.jobs_lock:
                if job_id not in self.pending_jobs:
                    continue
                job = self.pending_jobs[job_id]
                
            if side == 'front':
                job['front_outputs'] = outputs
            else:
                job['back_outputs'] = outputs
                
            if job['front_outputs'] is not None and job['back_outputs'] is not None:
                # Both sides are ready!
                front_det = self._decode_outputs_raw(job['front_outputs'])
                back_det = self._decode_outputs_raw(job['back_outputs'])
                
                front_img = cv2.cvtColor(job['front_bev'][:, :, 0], cv2.COLOR_GRAY2BGR)
                back_img = cv2.cvtColor(job['back_bev'][:, :, 0], cv2.COLOR_GRAY2BGR)
                
                draw_bev(front_img, front_det)
                draw_bev(back_img, back_det)
                
                img = job['img']
                show_rgb_image_with_boxes(img, front_det, self.calib, kFrontBoundary)
                
                self.frame_count += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.avg_fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                
                cv2.putText(img, f"Python Backend - Async FPS: {self.avg_fps:.1f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                if not self.result_queue.full():
                    self.result_queue.put((img, front_img, back_img))
                    
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
        
        self.lbl_full = QLabel()
        self.lbl_full.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_full)
        
        if full_screen:
            self.showFullScreen()
        else:
            self.resize(1216, 975)
            self.show()
            
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16) # ~60fps UI refresh
        
    def update_frame(self):
        result = self.app_model.get_latest_result()
        if result is None:
            return
            
        img, front, back = result
        
        # Rotate to match C++
        front_rot = cv2.rotate(front, cv2.ROTATE_90_COUNTERCLOCKWISE)
        back_rot = cv2.rotate(back, cv2.ROTATE_90_CLOCKWISE)
        
        # Construct the layout: [Back BEV | Front BEV] vertically stacked under the RGB image
        bev_concat = cv2.hconcat([back_rot, front_rot])
        
        # The RGB image needs to be the same width as bev_concat
        target_width = bev_concat.shape[1]
        target_height = int(img.shape[0] * (target_width / img.shape[1]))
        img_resized = cv2.resize(img, (target_width, target_height))

        full_img = cv2.vconcat([img_resized, bev_concat])

        lbl_w, lbl_h = self.lbl_full.width(), self.lbl_full.height()
        if lbl_w > 0 and lbl_h > 0:
            aspect_ratio = full_img.shape[1] / full_img.shape[0]
            if lbl_w / lbl_h > aspect_ratio:
                new_h = lbl_h
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = lbl_w
                new_h = int(new_w / aspect_ratio)
            full_img = cv2.resize(full_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.lbl_full.setPixmap(QPixmap.fromImage(qimg))

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
    parser.add_argument("--backend", type=str, default="dxnn")
    parser.add_argument("--full_screen", action="store_true")
    parser.add_argument("--exit-btn", action="store_true")
    args = parser.parse_args()

    # The dataset path is implicitly expected from the workspace layout
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../cpp/sfa3d/dataset/kitti/demo/2011_09_26_drive_0014_sync/2011_09_26/2011_09_26_drive_0014_sync'))
    
    app_model = SFA3DApp(args.model, dataset_dir)
    
    app = QApplication(sys.argv)
    window = DemoWindow(app_model, args.full_screen)
    app.exec_()

if __name__ == "__main__":
    main()
