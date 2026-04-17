import os
import cv2
import time
import argparse
import numpy as np
import threading
import queue
from dx_engine import InferenceEngine

from pathlib import Path

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

# 전역 변수 및 동기화 설정
result_queue = queue.Queue(maxsize=10) # 추론 결과 데이터를 담는 큐
callback_lock = threading.Lock()

class AsyncDepthAnything:
    def __init__(self, model_path):
        print(f"Initializing DX Engine (Async Mode) with Depth Anything...")
        self.engine = InferenceEngine(model_path)

        # 모델 입력 정보 가져오기 (NCHW 또는 NHWC 대응)
        input_info = self.engine.get_input_tensors_info()[0]["shape"]
        # 샘플 코드 1번의 형식을 따름: [Batch, Channel, Height, Width] (NCHW)
        _, _, self.net_h, self.net_w = input_info
        print(f"Model Input Size: {self.net_w}x{self.net_h}")

        # 콜백 함수 등록 [cite: 2682, 4973]
        self.engine.register_callback(self.inference_callback)

        # 정규화 파라미터
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, frame):
        """이미지 전처리: BGR->RGB, Resize, Normalize, NCHW 변환"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img_rgb, (self.net_w, self.net_h))

        x = img_input.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, [2, 0, 1]) # HWC -> CHW
        x = np.expand_dims(x, 0).astype(np.float32)
        return x

    def inference_callback(self, ie_outputs, user_args):
        """추론 완료 시 호출되는 콜백 함수 [cite: 2675, 4094]"""
        # user_args에는 원본 프레임이 들어있음
        original_frame = user_args
        prediction = ie_outputs[0]

        with callback_lock:
            # 처리 속도가 큐를 넣는 속도보다 느릴 경우를 대비해 가장 오래된 결과 제거
            if result_queue.full():
                result_queue.get()
            # 원본 이미지와 추론 결과(Depth Map)를 매칭하여 큐에 삽입
            result_queue.put((original_frame, prediction))
        return 0

    def run_async(self, frame):
        """비동기 추론 요청 호출 [cite: 2666, 4791]"""
        input_tensor = self.preprocess(frame)
        # run_async(입력데이터, 콜백으로 전달할 유저인자)
        # 여기서 user_args로 원본 프레임을 전달하여 결과와 화면을 매칭함
        self.engine.run_async(input_tensor, user_arg=frame)

    def close(self):
        """엔진 스레드/리소스 정리 (종료 시 호출)."""
        for name in ("release", "destroy", "close", "shutdown"):
            fn = getattr(self.engine, name, None)
            if callable(fn):
                try:
                    fn()
                    break
                except Exception:
                    pass

def create_depth_map(depth, grayscale=False):
    """깊이 데이터를 시각화 가능한 컬러맵으로 변환"""
    depth = np.squeeze(depth)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    # Min-Max 정규화 (0-255)
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 0:
        normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = np.zeros_like(depth)

    normalized_depth = normalized_depth.astype(np.uint8)

    if grayscale:
        return cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)
    else:
        return cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO)

def _parse_bg_color_rgb(s: str) -> tuple:
    """R,G,B (0-255) 문자열 → OpenCV용 BGR 튜플."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected R,G,B e.g. 0,0,0 or 40,40,45")
    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise argparse.ArgumentTypeError("each value must be 0-255")
    return (b, g, r)

def _screen_wh():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(w), int(h)
    except Exception:
        return 1920, 1080

def _letterbox_to_screen(img, sw, sh, bg_bgr):
    """비율 유지로 스케일 후 (sw,sh) 캔버스 가운데 배치; 여백은 bg_bgr."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        canvas = np.empty((sh, sw, 3), dtype=np.uint8)
        canvas[:] = bg_bgr
        return canvas
    scale = min(sw / w, sh / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.empty((sh, sw, 3), dtype=np.uint8)
    canvas[:] = bg_bgr
    x0 = (sw - nw) // 2
    y0 = (sh - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas

def _draw_fps_overlay(bgr: np.ndarray, fps: float) -> None:
    """반투명 패널 + 외곽선 텍스트로 FPS 표시 (원본 이미지에 in-place)."""
    text = f"{fps:.1f} FPS"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.08  # 0.72 * 1.5
    thick = 3
    pad_x, pad_y = 24, 17
    margin = 15
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    box_w = tw + pad_x * 2
    box_h = th + bl + pad_y * 2
    h_img, w_img = bgr.shape[:2]
    x0 = min(margin, max(0, w_img - box_w - margin))
    y0 = min(margin, max(0, h_img - box_h - margin))
    x1 = min(x0 + box_w, w_img)
    y1 = min(y0 + box_h, h_img)
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return
    roi = bgr[y0:y1, x0:x1]
    panel = np.full_like(roi, (38, 40, 44), dtype=np.uint8)
    blended = cv2.addWeighted(roi, 0.52, panel, 0.48, 0)
    bgr[y0:y1, x0:x1] = blended
    stripe_w = min(6, x1 - x0)
    if stripe_w > 0:
        st = bgr[y0:y1, x0 : x0 + stripe_w].astype(np.float32)
        accent = np.array([[[92, 168, 255]]], dtype=np.float32)  # BGR warm accent
        bgr[y0:y1, x0 : x0 + stripe_w] = (
            st * 0.35 + accent * 0.65
        ).astype(np.uint8)
    cv2.rectangle(bgr, (x0, y0), (x1 - 1, y1 - 1), (96, 102, 110), 1, cv2.LINE_AA)
    tx = x0 + pad_x
    ty = y0 + pad_y + th
    for ox, oy in (
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    ):
        cv2.putText(
            bgr,
            text,
            (tx + ox, ty + oy),
            font,
            scale,
            (0, 0, 0),
            thick + 1,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr,
        text,
        (tx, ty),
        font,
        scale,
        (236, 240, 245),
        thick,
        cv2.LINE_AA,
    )


def _draw_model_name_overlay(bgr: np.ndarray, model_path: str) -> None:
    """화면 맨 위 중앙에 사용 중인 모델 파일명 표시."""
    name = os.path.basename(model_path)
    text = f"Model: {name}"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.93  # 0.62 * 1.5
    thick = 2
    pad_x, pad_y = 21, 14
    margin = 12
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    box_w = tw + pad_x * 2
    box_h = th + bl + pad_y * 2
    h_img, w_img = bgr.shape[:2]
    x0 = max(margin, (w_img - box_w) // 2)
    y0 = margin
    x1 = min(x0 + box_w, w_img)
    y1 = min(y0 + box_h, h_img)
    if x1 <= x0 + 4 or y1 <= y0 + 4:
        return
    roi = bgr[y0:y1, x0:x1]
    panel = np.full_like(roi, (32, 34, 38), dtype=np.uint8)
    bgr[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.45, panel, 0.55, 0)
    cv2.rectangle(bgr, (x0, y0), (x1 - 1, y1 - 1), (80, 86, 94), 1, cv2.LINE_AA)
    tx = x0 + pad_x
    ty = y0 + pad_y + th
    for ox, oy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1)):
        cv2.putText(
            bgr,
            text,
            (tx + ox, ty + oy),
            font,
            scale,
            (0, 0, 0),
            thick + 1,
            cv2.LINE_AA,
        )
    cv2.putText(
        bgr,
        text,
        (tx, ty),
        font,
        scale,
        (220, 224, 230),
        thick,
        cv2.LINE_AA,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="models/depth_anything_v2.dxnn",
        help="Path to .dxnn model",
    )
    parser.add_argument(
        "-s", "--side", action="store_true", help="Show side by side"
    )
    parser.add_argument(
        "-g", "--grayscale", action="store_true", help="Use grayscale colormap"
    )
    parser.add_argument(
        "--bg-color",
        default="0,0,0",
        type=_parse_bg_color_rgb,
        dest="margin_bgr",
        metavar="R,G,B",
        help="Fullscreen letterbox margin color in RGB 0-255 (default: 0,0,0 black)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    async_depth = AsyncDepthAnything(args.model)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    screen_w, screen_h = _screen_wh()

    # 추론 FPS: 최근 5초 구간의 결과 개수 / 5, 1초마다 갱신
    result_timestamps = []  # inference 결과가 나온 시각들
    fps_display = 0.0
    last_fps_update = time.time()
    FPS_WINDOW_SEC = 5.0
    FPS_UPDATE_INTERVAL = 1.0

    print("Start Async Processing. Press 'q' to exit.")

    win_name = "Depth Anything v2 Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    window_fullscreen = False
    launcher_ready = False

    try:
        while True:
            now = time.time()
            # 1초마다: 최근 5초 구간 결과 개수로 FPS 갱신
            if now - last_fps_update >= FPS_UPDATE_INTERVAL:
                cut = now - FPS_WINDOW_SEC
                result_timestamps[:] = [t for t in result_timestamps if t >= cut]
                fps_display = len(result_timestamps) / FPS_WINDOW_SEC
                last_fps_update = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = cv2.VideoCapture(args.video)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, frame = cap.read()
                if not ret:
                    print("Error: Could not rewind or reopen video for loop.")
                    break

            # 1. 비동기 추론 요청 (Non-blocking)
            async_depth.run_async(frame)

            # 2. 큐에서 완료된 결과 확인 (추론 완료된 데이터가 있을 때만 렌더링)
            if not result_queue.empty():
                orig, pred = result_queue.get()
                result_timestamps.append(time.time())

                # 결과 처리 (ColorMap 적용)
                depth_colored = create_depth_map(pred, args.grayscale)

                # -s: 깊이 시각화를 카메라 입력과 동일 (W×H)로 맞춰 원본 옆에 붙임
                if args.side:
                    oh, ow = orig.shape[:2]
                    depth_scaled = cv2.resize(
                        depth_colored,
                        (ow, oh),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    display_content = np.concatenate((orig, depth_scaled), axis=1)
                else:
                    display_content = depth_colored

                frame_show = _letterbox_to_screen(
                    display_content, screen_w, screen_h, args.margin_bgr
                )
                _draw_fps_overlay(frame_show, fps_display)
                _draw_model_name_overlay(frame_show, args.model)
                cv2.imshow(win_name, frame_show)
                if not window_fullscreen:
                    cv2.setWindowProperty(
                        win_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN,
                    )
                    window_fullscreen = True

                if not launcher_ready:
                    notify_launcher_ready()
                    launcher_ready = True

            # 'q' or 'ESC' 키를 누르면 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                time.sleep(0.5)
                break

    finally:
        try:
            async_depth.close()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("\nFinished.")

if __name__ == "__main__":
    main()
