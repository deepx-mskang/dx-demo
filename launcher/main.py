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
    QComboBox,
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
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000

LANGUAGE_OPTIONS = [
    ("English", "en"),
    ("中文", "zh"),
    ("日本語", "ja"),
    ("한국어", "ko"),
]

MAIN_TITLE_I18N = {
    "en": "DEEPX AI Demo Hub",
    "zh": "DEEPX AI 演示中心",
    "ja": "DEEPX AI デモハブ",
    "ko": "DEEPX AI 데모 허브",
}

# How many launcher cards to show (first N entries of LAUNCHER_ITEMS).
NUM_ITEMS = 12

# Grid columns; rows are computed as ceil(NUM_ITEMS / GRID_COLUMNS).
GRID_COLUMNS = 4

# Launcher root (directory containing this file); assets and ready state live here.
_ROOT = Path(__file__).resolve().parent

# Wait ends early when the launched app updates this file (path passed in $DX_LAUNCHER_READY_FILE).
READY_STATUS_DIR = (_ROOT / "ready").resolve()
READY_FILE_ENV = "DX_LAUNCHER_READY_FILE"

_ready_watcher: QFileSystemWatcher | None = None
_ready_finishers: dict[str, Callable[[], None]] = {}

# Per-item configuration. Paths in "image" are relative to _ROOT unless absolute.
# video_script / camera_script (optional): resolved relative to _ROOT unless absolute.
#   Omit one to show a single button (e.g. Drone Tracking uses video_script only).
# video_label / camera_label (optional): override button text (default "Video" / "Camera"),
#   e.g. Performance Monitoring uses "Start" / "Stop" with the same script keys.
# extra_buttons (optional): extra actions for some demos only, e.g.
#   "extra_buttons": [{"label": "3D View", "script": "../scripts/demo_3d.sh", "loading_sec": 15}, ...]
# loading_sec (optional, seconds): card-wide default for all buttons on this item.
# video_loading_sec / camera_loading_sec (optional): override per primary/secondary button.
# Early end of Wait: touch or write $DX_LAUNCHER_READY_FILE (under READY_STATUS_DIR per click).
DEFAULT_BUTTON_LOADING_SEC = 1.0
LAUNCHER_ITEMS = [
    {
        "title": "Hyundai Robotics - Delivery Robot (DAL-e)",
        "title_i18n": {
            "zh": "Hyundai Robotics - 配送机器人 (DAL-e)",
            "ja": "Hyundai Robotics - 配送ロボット (DAL-e)",
            "ko": "현대로보틱스 - 배송 로봇 (DAL-e)",
        },
        "image": "assets/demo-robotics.png",
        "video_script": "../scripts/run_robotics_video.sh",
        "camera_script": "../scripts/run_robotics.sh",
        "loading_sec": 5,
    },
    {
        "title": "YOLO26-S (OD / POSE / SEG)",
        "title_i18n": {
            "zh": "YOLO26-S (目标检测 / 姿态 / 分割)",
            "ja": "YOLO26-S (物体検出 / ポーズ / 分割)",
            "ko": "YOLO26-S (객체 탐지 / 자세 / 분할)",
        },
        "image": "assets/demo-yolo26.png",
        "video_script": "../scripts/run_yolo26_4_video.sh",
        "camera_script": "../scripts/run_yolo26_4.sh",
        "loading_sec": 5,
    },
    {
        "title": "Mono Depth Estimation (Depth Anything v2)",
        "title_i18n": {
            "zh": "单目深度估计 (Depth Anything v2)",
            "ja": "単眼深度推定 (Depth Anything v2)",
            "ko": "단안 깊이 추정 (Depth Anything v2)",
        },
        "image": "assets/demo-depth.png",
        "video_script": "../scripts/run_depth_video.sh",
        "camera_script": "../scripts/run_depth.sh",
        "camera_label": "Cam (518)",
        "extra_buttons": [{"label": "Cam (224)", "script": "../scripts/run_depth_224.sh"}],
        "loading_sec": 5,
    },
    {
        "title": "PaddleOCR v5",
        "title_i18n": {
            "zh": "PaddleOCR v5",
            "ja": "PaddleOCR v5",
            "ko": "PaddleOCR v5",
        },
        "image": "assets/demo-ocr.gif",
        "video_script": "../scripts/run_ocr_web.sh",
        "camera_script": "../scripts/run_ocr.sh",
        "loading_sec": 30,
        "video_label": "Web Based",
    },
    {
        "title": "YOLOv5S Multi-channel (36)",
        "title_i18n": {
            "zh": "YOLOv5S 多通道 (36)",
            "ja": "YOLOv5S マルチチャンネル (36)",
            "ko": "YOLOv5S 멀티채널 (36)",
        },
        "image": "assets/demo-yolo-multi.png",
        "video_script": "../scripts/run_yolo_multi_video.sh",
        "camera_script": "../scripts/run_yolo_multi.sh",
        "loading_sec": 30,
    },
    {
        "title": "Hand Landmark & Pose Estimation",
        "title_i18n": {
            "zh": "手部关键点与姿态估计",
            "ja": "手のランドマーク・姿勢推定",
            "ko": "손 랜드마크 및 자세 추정",
        },
        "image": "assets/demo-hands.png",
        "video_script": "../scripts/run_hands_video.sh",
        "camera_script": "../scripts/run_hands.sh",
        "loading_sec": 5,
    },
    {
        "title": "Real-Time Road Scene Perception",
        "title_i18n": {
            "zh": "实时道路场景感知",
            "ja": "リアルタイム道路シーン認識",
            "ko": "실시간 도로 장면 인식",
        },
        "image": "assets/demo-automotive.png",
        "video_script": "../scripts/run_PIDNet.sh",
        "camera_script": "../scripts/run_YOLOPv2.sh",
        "loading_sec": 5,
    },
    {
        "title": "Drone Tracking",
        "title_i18n": {
            "zh": "Drone Tracking",
            "ja": "Drone Tracking",
            "ko": "Drone Tracking",
        },
        "image": "assets/demo-dron.png",
        "video_script": "../scripts/./run_drone_1.sh",
        "video_label": "Video",
        "loading_sec": 5,
    },
    {
        "title": "DEEPX Model Zoo",
        "title_i18n": {
            "zh": "DEEPX 模型库",
            "ja": "DEEPX モデル集",
            "ko": "DEEPX 모델 저장소",
        },
        "image": "assets/demo-modelzoo.png",
        "video_label": "Open",
        "camera_label": "Close",
        "video_script": "../scripts/run_modelzoo.sh",
        "camera_script": "../scripts/kill_modelzoo.sh",
    },
    {
        "title": "CLIP Single-channel",
        "title_i18n": {
            "zh": "CLIP 单通道",
            "ja": "CLIP シングルチャンネル",
            "ko": "CLIP 단일 채널",
        },
        "image": "assets/demo-clip-single.png",
        "video_script": "../scripts/run_clip_single_video.sh",
        "camera_script": "../scripts/run_clip_single.sh",
        "loading_sec": 15,
    },
    {
        "title": "CLIP Multi-channel",
        "title_i18n": {
            "zh": "CLIP 多通道",
            "ja": "CLIP マルチチャンネル",
            "ko": "CLIP 멀티채널",
        },
        "image": "assets/demo-clip.png",
        "video_script": "../scripts/run_clip_video.sh",
        "camera_script": "../scripts/run_clip.sh",
        "loading_sec": 30,
        # "extra_buttons": [{"label": "Export", "script": "../scripts/export.sh"}],
        # "loading_sec": 12,           # 이 카드는 기본 12초
        # "video_loading_sec": 20,     # Video만 20초
    },
    {
        "title": "Performance Monitoring (CPU / NPU)",
        "title_i18n": {
            "zh": "性能监控 (CPU / NPU)",
            "ja": "性能モニタリング (CPU / NPU)",
            "ko": "성능 모니터링 (CPU / NPU)",
        },
        "image": "assets/demo-perf.png",
        "video_label": "Start",
        "camera_label": "Stop",
        "video_script": "../scripts/run_dxtop.sh",
        "camera_script": "../scripts/kill_perf.sh",
        "camera_loading_sec": 0,
        "extra_buttons": [{"label": "Kill All", "script": "../scripts/kill_all.sh"}],
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


def _localized_main_title(language_code: str) -> str:
    return MAIN_TITLE_I18N.get(language_code, MAIN_TITLE_I18N["en"])


def _localized_item_title(cfg: dict, language_code: str) -> str:
    fallback = str(cfg["title"])
    if language_code == "en":
        return fallback
    translations = cfg.get("title_i18n")
    if isinstance(translations, dict):
        translated = translations.get(language_code)
        if translated:
            return str(translated)
    return fallback


def _ensure_ready_status_dir() -> Path:
    READY_STATUS_DIR.mkdir(parents=True, exist_ok=True)
    return READY_STATUS_DIR


def _resolve_shell_script(script_path: str) -> Path | None:
    expanded = Path(os.path.expanduser(script_path))
    if not str(expanded):
        return None
    p = expanded if expanded.is_absolute() else (_ROOT / expanded).resolve()
    if not p.is_file():
        print(f"[launcher] script not found: {p}", file=sys.stderr)
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

    def set_title(self, title: str) -> None:
        self._title.setText(title)

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
        self.setWindowTitle("Launcher")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.setStyleSheet("background-color: #000000;")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        header = QGridLayout()
        header.setContentsMargins(4, 0, 4, 0)
        header.setSpacing(10)

        self._main_title = QLabel(_localized_main_title("en"))
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        self._main_title.setFont(title_font)
        self._main_title.setAlignment(Qt.AlignCenter)
        self._main_title.setStyleSheet("color: #ffffff; background: transparent;")

        language_controls = QWidget()
        language_row = QHBoxLayout(language_controls)
        language_row.setContentsMargins(0, 0, 0, 0)
        language_row.setSpacing(10)

        language_label = QLabel("Language")
        language_label.setStyleSheet(
            "color: #c5f3ff; background: transparent; font-size: 13px;"
        )

        self._language_combo = QComboBox()
        for label, code in LANGUAGE_OPTIONS:
            self._language_combo.addItem(label, code)
        self._language_combo.setCurrentIndex(0)
        self._language_combo.setCursor(Qt.PointingHandCursor)
        self._language_combo.setFixedWidth(160)
        self._language_combo.setStyleSheet(
            "QComboBox {"
            " color: #ffffff;"
            " background-color: #0a1f33;"
            " border: 1px solid #00b4d8;"
            " border-radius: 4px;"
            " padding: 6px 12px;"
            " font-size: 13px;"
            "}"
            "QComboBox:hover {"
            " border-color: #48e1ff;"
            "}"
            "QComboBox::drop-down {"
            " width: 28px;"
            " border: none;"
            "}"
            "QComboBox QAbstractItemView {"
            " color: #ffffff;"
            " background-color: #0a1f33;"
            " selection-background-color: #123a5c;"
            " selection-color: #ffffff;"
            " outline: none;"
            "}"
        )

        language_row.addWidget(language_label)
        language_row.addWidget(self._language_combo)

        left_balance = QWidget()
        left_balance.setFixedWidth(language_controls.sizeHint().width())

        header.addWidget(left_balance, 0, 0)
        header.addWidget(self._main_title, 0, 1, alignment=Qt.AlignCenter)
        header.addWidget(language_controls, 0, 2, alignment=Qt.AlignRight)
        header.setColumnStretch(1, 1)
        root.addLayout(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        root.addLayout(grid, stretch=1)

        n = min(NUM_ITEMS, len(LAUNCHER_ITEMS))
        if NUM_ITEMS > len(LAUNCHER_ITEMS):
            print(
                f"[launcher] NUM_ITEMS ({NUM_ITEMS}) > len(LAUNCHER_ITEMS) "
                f"({len(LAUNCHER_ITEMS)}); showing {n} items.",
                file=sys.stderr,
            )

        self._launcher_items: list[tuple[LauncherItem, dict]] = []
        for i in range(n):
            cfg = LAUNCHER_ITEMS[i]
            img_path = _resolve_image_path(str(cfg["image"]))
            fb = _fallback_loading_ms(cfg)
            v_label = cfg.get("video_label")
            c_label = cfg.get("camera_label")
            video_ms = _button_loading_ms(cfg, "video_loading_sec", fb)
            camera_ms = _button_loading_ms(cfg, "camera_loading_sec", fb)
            extras = _normalize_extra_buttons(cfg.get("extra_buttons"), fb)
            actions: list[tuple[str, str, int]] = []
            video_script = cfg.get("video_script")
            if video_script:
                actions.append(
                    (
                        str(v_label) if v_label is not None else "Video",
                        str(video_script),
                        video_ms,
                    )
                )
            camera_script = cfg.get("camera_script")
            if camera_script:
                actions.append(
                    (
                        str(c_label) if c_label is not None else "Camera",
                        str(camera_script),
                        camera_ms,
                    )
                )
            actions.extend(extras)
            item = LauncherItem(
                title=_localized_item_title(cfg, "en"),
                image_path=img_path,
                actions=actions,
            )
            self._launcher_items.append((item, cfg))
            row, col = i // GRID_COLUMNS, i % GRID_COLUMNS
            grid.addWidget(item, row, col)

        self._language_combo.currentIndexChanged.connect(self._apply_language)
        self._apply_language()
        self._place_center_on_screen()

    def _current_language_code(self) -> str:
        code = self._language_combo.currentData()
        return str(code) if code else "en"

    def _apply_language(self, _index: int | None = None) -> None:
        language_code = self._current_language_code()
        self._main_title.setText(_localized_main_title(language_code))
        for item, cfg in self._launcher_items:
            item.set_title(_localized_item_title(cfg, language_code))

    def _place_center_on_screen(self) -> None:
        """Center the window on the primary screen's available (work) area."""
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + max(0, (geo.width() - self.width()) // 2)
        y = geo.y() + 10#max(0, (geo.height() - self.height()) // 2)
        self.move(x, y)


def main() -> int:
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
