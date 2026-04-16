#!/usr/bin/env python3
"""
DEEPX demo application launcher (PyQt5).
Adjust NUM_ITEMS and LAUNCHER_ITEMS at the top to add or reconfigure demos.
"""

from __future__ import annotations

import os
from collections.abc import Callable
import subprocess
import sys
import uuid
from pathlib import Path

from PyQt5.QtCore import QFileSystemWatcher, Qt, QTimer
from PyQt5.QtGui import QFont, QGuiApplication, QPixmap, QShowEvent
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# --- Window geometry ---
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# How many launcher cards to show (first N entries of LAUNCHER_ITEMS).
NUM_ITEMS = 9

# Grid columns; rows are computed as ceil(NUM_ITEMS / GRID_COLUMNS).
GRID_COLUMNS = 3

# Project root (directory containing this file); images are resolved relative to it.
_ROOT = Path(__file__).resolve().parent

# Wait ends early when the launched app updates this file (path passed in $DX_LAUNCHER_READY_FILE).
READY_STATUS_DIR = Path(os.path.expanduser("~/demos/dx-demo-launcher/ready")).resolve()
READY_FILE_ENV = "DX_LAUNCHER_READY_FILE"

_ready_watcher: QFileSystemWatcher | None = None
_ready_finishers: dict[str, Callable[[], None]] = {}

# Per-item configuration. Paths in "image" are relative to _ROOT unless absolute.
# video_script / camera_script: passed to bash (expanduser applied).
# video_label / camera_label (optional): override button text (default "Video" / "Camera"),
#   e.g. Performance Monitoring uses "Start" / "Stop" with the same script keys.
# extra_buttons (optional): extra actions for some demos only, e.g.
#   "extra_buttons": [{"label": "3D View", "script": "~/scripts/demo_3d.sh", "loading_sec": 15}, ...]
# loading_sec (optional, seconds): card-wide default for all buttons on this item.
# video_loading_sec / camera_loading_sec (optional): override per primary/secondary button.
# Early end of Wait: touch or write $DX_LAUNCHER_READY_FILE (under READY_STATUS_DIR per click).
DEFAULT_BUTTON_LOADING_SEC = 1.0
LAUNCHER_ITEMS = [
    {
        "title": "Hyundai Robotics - Delivery Robot (MobED)",
        "image": "assets/demo-robotics.png",
        "video_script": "~/scripts/run_robotics_video.sh",
        "camera_script": "~/scripts/run_robotics.sh",
        "loading_sec": 5,
    },
    {
        "title": "YOLO26-S (OD / POSE / SEG / CLS)",
        "image": "assets/demo-yolo26.png",
        "video_script": "~/scripts/run_yolo26_4_video.sh",
        "camera_script": "~/scripts/run_yolo26_4.sh",
        "loading_sec": 5,
    },
    {
        "title": "Mono Depth Estimation (Depth Anything v2)",
        "image": "assets/demo-depth.png",
        "video_script": "~/scripts/run_depth_video.sh",
        "camera_script": "~/scripts/run_depth.sh",
        "loading_sec": 5,
    },
    {
        "title": "PaddleOCR v5",
        "image": "assets/demo-ocr.gif",
        "video_script": "~/scripts/run_ocr.sh",
        "camera_script": "~/scripts/run_ocr_autofocus_dis.sh",
        "loading_sec": 20,
        "video_label": "Run /AF-En",
        "camera_label": "Run /AF-Dis",
    },
    {
        "title": "CLIP Single-channel",
        "image": "assets/demo-clip-single.png",
        "video_script": "~/scripts/run_clip_single_video.sh",
        "camera_script": "~/scripts/run_clip_single.sh",
        "loading_sec": 15,
    },
    {
        "title": "YOLOv5S Multi-channel (36)",
        "image": "assets/demo-yolo-multi.png",
        "video_script": "~/scripts/run_yolo_multi_video.sh",
        "camera_script": "~/scripts/run_yolo_multi.sh",
        "loading_sec": 30,
    },
    {
        "title": "DEEPX Model Zoo",
        "image": "assets/demo-modelzoo.png",
        "video_label": "Open",
        "camera_label": "Close",
        "video_script": "~/scripts/run_modelzoo.sh",
        "camera_script": "~/scripts/kill_modelzoo.sh",
    },
    {
        "title": "CLIP Multi-channel",
        "image": "assets/demo-clip.png",
        "video_script": "~/scripts/run_clip_video.sh",
        "camera_script": "~/scripts/run_clip.sh",
        "loading_sec": 30,
        # "extra_buttons": [{"label": "Export", "script": "~/scripts/export.sh"}],
        # "loading_sec": 12,           # 이 카드는 기본 12초
        # "video_loading_sec": 20,     # Video만 20초
    },
    {
        "title": "Performance Monitoring (CPU / NPU)",
        "image": "assets/demo-perf.png",
        "video_label": "Start",
        "camera_label": "Stop",
        "video_script": "~/scripts/run_dxtop.sh",
        "camera_script": "~/scripts/kill_perf.sh",
        "camera_loading_sec": 0,
    },
]


def _resolve_image_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (_ROOT / p)


def _fallback_loading_ms(cfg: dict) -> int:
    v = cfg.get("loading_sec")
    if v is not None:
        try:
            return max(0, int(float(v) * 1000))
        except (TypeError, ValueError):
            pass
    return max(0, int(DEFAULT_BUTTON_LOADING_SEC * 1000))


def _button_loading_ms(cfg: dict, key: str, fallback_ms: int) -> int:
    v = cfg.get(key)
    if v is not None:
        try:
            return max(0, int(float(v) * 1000))
        except (TypeError, ValueError):
            pass
    return fallback_ms


def _normalize_extra_buttons(raw: object, fallback_ms: int) -> list[tuple[str, str, int]]:
    """Parse optional extra_buttons: (label, script, loading_ms)."""
    if not raw:
        return []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[tuple[str, str, int]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        script = entry.get("script")
        if label is None or script is None or not str(label).strip():
            continue
        ls = entry.get("loading_sec")
        if ls is not None:
            try:
                ms = max(0, int(float(ls) * 1000))
            except (TypeError, ValueError):
                ms = fallback_ms
        else:
            ms = fallback_ms
        out.append((str(label), str(script), ms))
    return out


def _ensure_ready_status_dir() -> Path:
    READY_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    return READY_STATUS_DIR


def _resolve_shell_script(script_path: str) -> Path | None:
    expanded = os.path.expanduser(script_path)
    if not expanded:
        return None
    p = Path(expanded)
    if not p.is_file():
        print(f"[launcher] script not found: {expanded}", file=sys.stderr)
        return None
    return p


def _run_shell_script(
    script_path: str,
    extra_env: dict[str, str] | None = None,
) -> bool:
    p = _resolve_shell_script(script_path)
    if p is None:
        return False
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        subprocess.Popen(
            ["/bin/bash", str(p)],
            cwd=str(p.parent),
            start_new_session=True,
            env=env,
        )
    except OSError as e:
        print(f"[launcher] failed to run {p}: {e}", file=sys.stderr)
        return False
    return True


def _on_ready_file_changed(path: str) -> None:
    key = str(Path(path).resolve())
    finisher = _ready_finishers.get(key)
    if finisher is not None:
        finisher()


def _ensure_ready_watcher() -> QFileSystemWatcher:
    global _ready_watcher
    if _ready_watcher is None:
        parent = QApplication.instance()
        _ready_watcher = QFileSystemWatcher(parent)
        _ready_watcher.fileChanged.connect(_on_ready_file_changed)
    return _ready_watcher


def _teardown_ready_watcher_if_idle() -> None:
    global _ready_watcher
    if _ready_watcher is None:
        return
    if _ready_finishers:
        return
    if _ready_watcher.files():
        return
    _ready_watcher.deleteLater()
    _ready_watcher = None


def _safe_restore_button(btn: QPushButton, text: str, normal_qss: str) -> None:
    try:
        btn.setText(text)
        btn.setStyleSheet(normal_qss)
        btn.setEnabled(True)
    except RuntimeError:
        pass


def _safe_set_button_enabled(btn: QPushButton, enabled: bool) -> None:
    try:
        btn.setEnabled(enabled)
    except RuntimeError:
        pass


class LauncherItem(QWidget):
    """One card: white panel (title + image) and primary/secondary action buttons (+ optional extras)."""

    def __init__(
        self,
        title: str,
        image_path: Path,
        actions: list[tuple[str, str, int]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._source_pixmap: QPixmap | None = None

        if image_path.is_file():
            self._source_pixmap = QPixmap(str(image_path))
        else:
            self._source_pixmap = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        self._white = QFrame()
        self._white.setObjectName("cardWhite")
        self._white.setStyleSheet(
            "#cardWhite { background-color: #ffffff; border: none; border-radius: 4px; }"
        )
        white_lay = QVBoxLayout(self._white)
        white_lay.setContentsMargins(12, 10, 12, 10)
        white_lay.setSpacing(8)

        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setWordWrap(True)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self._title.setFont(title_font)
        self._title.setStyleSheet("color: #000000; background: transparent;")
        white_lay.addWidget(self._title)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setMinimumHeight(80)
        self._image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._image_label.setStyleSheet("background: transparent;")
        white_lay.addWidget(self._image_label, stretch=1)

        outer.addWidget(self._white, stretch=1)

        _btn_qss = (
            "QPushButton {"
            " color: #c5f3ff;"
            " background-color: #0a1f33;"
            " border: 1px solid #00b4d8;"
            " border-radius: 4px;"
            " font-size: 13px;"
            " padding: 6px 14px;"
            "}"
            "QPushButton:hover {"
            " background-color: #123a5c;"
            " border-color: #48e1ff;"
            " color: #ffffff;"
            "}"
            "QPushButton:pressed {"
            " background-color: #061525;"
            " border-color: #0096b4;"
            "}"
        )
        _btn_qss_wait = (
            "QPushButton, QPushButton:disabled {"
            " color: #fff3d4;"
            " background-color: #3a3010;"
            " border: 1px solid #d9a012;"
            " border-radius: 4px;"
            " font-size: 13px;"
            " padding: 6px 14px;"
            "}"
        )

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(4, 0, 4, 0)
        btn_row.setSpacing(10)
        action_buttons: list[QPushButton] = []
        action_meta: list[tuple[QPushButton, str]] = []
        for label, script, loading_ms in actions:
            btn = QPushButton(label)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(_btn_qss)

            def _make_handler(
                b: QPushButton,
                orig_lbl: str,
                scr: str,
                delay_ms: int,
            ):
                def _on_click(_checked: bool = False) -> None:
                    if _resolve_shell_script(scr) is None:
                        return

                    for other_btn in action_buttons:
                        _safe_set_button_enabled(other_btn, False)
                    b.setText("Wait")
                    b.setStyleSheet(_btn_qss_wait)

                    status_dir = _ensure_ready_status_dir()
                    ready_path = status_dir / f"{uuid.uuid4().hex}.ready"
                    ready_path.write_text("", encoding="utf-8")
                    key = str(ready_path.resolve())
                    state: dict[str, bool] = {"done": False}
                    t = QTimer(b)
                    t.setSingleShot(True)

                    def _restore_buttons() -> None:
                        for restore_btn, restore_label in action_meta:
                            _safe_restore_button(restore_btn, restore_label, _btn_qss)

                    def _finalize_wait() -> None:
                        if state["done"]:
                            return
                        state["done"] = True
                        _ready_finishers.pop(key, None)
                        try:
                            t.stop()
                        except RuntimeError:
                            pass
                        rw = _ready_watcher
                        if rw is not None:
                            try:
                                rw.removePath(key)
                            except (TypeError, RuntimeError):
                                pass
                        try:
                            Path(key).unlink(missing_ok=True)
                        except OSError:
                            pass
                        _restore_buttons()
                        _teardown_ready_watcher_if_idle()

                    _ready_finishers[key] = _finalize_wait
                    _ensure_ready_watcher().addPath(key)

                    if not _run_shell_script(scr, {READY_FILE_ENV: key}):
                        _finalize_wait()
                        return

                    t.timeout.connect(_finalize_wait)
                    t.start(delay_ms)

                return _on_click

            btn.clicked.connect(_make_handler(btn, label, script, loading_ms))
            action_buttons.append(btn)
            action_meta.append((btn, label))
        if action_buttons:
            _btn_w = max(b.sizeHint().width() for b in action_buttons)
            for b in action_buttons:
                b.setFixedWidth(_btn_w)
        btn_row.addStretch(1)
        for b in action_buttons:
            btn_row.addWidget(b, alignment=Qt.AlignRight | Qt.AlignVCenter)
        outer.addLayout(btn_row)

        self._refresh_image()

    def showEvent(self, event: QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._refresh_image()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_image()

    def _refresh_image(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            self._image_label.setText("No image")
            self._image_label.setStyleSheet(
                "background: transparent; color: #888888; font-size: 12px;"
            )
            return
        self._image_label.setText("")
        self._image_label.setStyleSheet("background: transparent;")
        # Available size inside the white frame for the image label
        self._image_label.updateGeometry()
        w = max(1, self._image_label.width())
        h = max(1, self._image_label.height())
        scaled = self._source_pixmap.scaled(
            w,
            h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DEEPX Demo Launcher")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet("background-color: #000000;")

        grid = QGridLayout(self)
        grid.setContentsMargins(16, 16, 16, 16)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        n = min(NUM_ITEMS, len(LAUNCHER_ITEMS))
        if NUM_ITEMS > len(LAUNCHER_ITEMS):
            print(
                f"[launcher] NUM_ITEMS ({NUM_ITEMS}) > len(LAUNCHER_ITEMS) "
                f"({len(LAUNCHER_ITEMS)}); showing {n} items.",
                file=sys.stderr,
            )

        for i in range(n):
            cfg = LAUNCHER_ITEMS[i]
            img_path = _resolve_image_path(str(cfg["image"]))
            fb = _fallback_loading_ms(cfg)
            v_label = cfg.get("video_label")
            c_label = cfg.get("camera_label")
            video_ms = _button_loading_ms(cfg, "video_loading_sec", fb)
            camera_ms = _button_loading_ms(cfg, "camera_loading_sec", fb)
            extras = _normalize_extra_buttons(cfg.get("extra_buttons"), fb)
            actions: list[tuple[str, str, int]] = [
                (
                    str(v_label) if v_label is not None else "Video",
                    str(cfg["video_script"]),
                    video_ms,
                ),
                (
                    str(c_label) if c_label is not None else "Camera",
                    str(cfg["camera_script"]),
                    camera_ms,
                ),
            ]
            actions.extend(extras)
            item = LauncherItem(
                title=str(cfg["title"]),
                image_path=img_path,
                actions=actions,
            )
            row, col = i // GRID_COLUMNS, i % GRID_COLUMNS
            grid.addWidget(item, row, col)

        self._place_center_on_screen()

    def _place_center_on_screen(self) -> None:
        """Center the window on the primary screen's available (work) area."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + max(0, (geo.width() - self.width()) // 2)
        y = geo.y() + 30#max(0, (geo.height() - self.height()) // 2)
        self.move(x, y)


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
