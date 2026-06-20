#!/usr/bin/env python3
"""
DEEPX 웹 히어로 톤의 Performance Monitor — 기능은 perf_monitor.py 와 동일.
(다크 베이스, 시안·전기 블루 글로우, 흰 타이포, 라운드 패널)

실행: .venv-perf/bin/python perf_monitor_design.py
"""

from __future__ import annotations

import sys
from typing import List, Optional, Tuple

import psutil
from PyQt5.QtCore import QEvent, QPoint, QRect, Qt, QTimer
from PyQt5.QtGui import (
    QColor,
    QKeySequence,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from perf_monitor import (
    NPU_CORE_COUNT,
    UPDATE_INTERVAL_MS,
    configure_overlay_window,
    fetch_npu_utilization,
    try_host_cpu_temperature,
    try_device_status_temperature,
)


class DX:
    """DEEPX hero: 딥 블랙, 전기 블루 → 시안, 퓨어 화이트."""
    #BG = "#000000"
    BG = "#1f1f1f"
    BG_PANEL = "#05080d"
    BG_PANEL_BORDER = "#0a1628"
    TEXT = "#ffffff"
    TEXT_DIM = "#a8b8cc"
    BLUE_DEEP = "#003d7a"
    BLUE_ELECTRIC = "#0078ff"
    CYAN = "#00e5ff"
    CYAN_SOFT = "#4df0ff"
    GLOW = "#00d4ff"
    # CPU 막대: 에메랄드 딥 → 비비드 그린 → 민트 하이라이트
    GREEN_DEEP = "#064e3b"
    GREEN_RICH = "#10b981"
    GREEN_BRIGHT = "#34d399"
    GREEN_GLOW = "#6ee7b7"


# NPU 코어 행과 동일한 라벨 스타일 (CPU 한 줄 행에도 사용)
ROW_LABEL_STYLE = (
    f"color: {DX.TEXT_DIM}; font-size: 11px; font-weight: 600; "
    f"background: transparent; border: none; min-width: 140px;"
)


class GlowProgressBar(QProgressBar):
    """블루→시안(NPU) 또는 에메랄드 녹색(CPU) 그라데이션 + 소프트 글로우."""

    def __init__(self, parent=None, *, cpu_style: bool = False):
        super().__init__(parent)
        self._cpu_style = cpu_style
        self.setTextVisible(True)
        self.setRange(0, 100)
        self.setFormat("%p%")
        self.setMinimumHeight(18)
        self.setMaximumHeight(18)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        r = self.rect().adjusted(2, 2, -2, -2)
        if self._cpu_style:
            track = QColor(14, 42, 30, 210)
        else:
            track = QColor(20, 40, 70, 200)
        painter.setPen(Qt.NoPen)
        painter.setBrush(track)
        painter.drawRoundedRect(r, 9, 9)

        pct = min(100, max(0, self.value()))
        if pct <= 0:
            if self.text():
                painter.setPen(QPen(QColor(DX.TEXT)))
                f = self.font()
                f.setPointSize(8)
                f.setBold(True)
                painter.setFont(f)
                painter.drawText(self.rect(), Qt.AlignCenter, self.text())
            return

        fw = max(3, int(r.width() * pct / 100.0))
        fr = r.adjusted(0, 0, fw - r.width(), 0)

        # 소프트 글로우 (막대 뒤)
        if self._cpu_style:
            glow_col = QColor(DX.GREEN_GLOW)
            glow_col.setAlpha(58)
        else:
            glow_col = QColor(0, 228, 255, 55)
        painter.setBrush(glow_col)
        painter.drawRoundedRect(fr.adjusted(-2, -1, 2, 1), 8, 8)

        g = QLinearGradient(fr.left(), 0, fr.right(), 0)
        if self._cpu_style:
            g.setColorAt(0.0, QColor(DX.GREEN_DEEP))
            g.setColorAt(0.42, QColor(DX.GREEN_RICH))
            g.setColorAt(1.0, QColor(DX.GREEN_BRIGHT))
        else:
            g.setColorAt(0.0, QColor(DX.BLUE_DEEP))
            g.setColorAt(0.45, QColor(DX.BLUE_ELECTRIC))
            g.setColorAt(1.0, QColor(DX.CYAN))
        painter.setBrush(g)
        painter.drawRoundedRect(fr, 7, 7)

        gloss_h = max(3, int(fr.height() * 0.42))
        top = QRect(fr.x() + 1, fr.y() + 1, fr.width() - 2, gloss_h)
        if top.height() > 2:
            hi = QLinearGradient(0, top.top(), 0, top.bottom())
            hi.setColorAt(0.0, QColor(255, 255, 255, 55))
            hi.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setBrush(hi)
            painter.drawRoundedRect(top, 5, 5)

        painter.setPen(QPen(QColor(DX.TEXT)))
        painter.setBrush(Qt.NoBrush)
        f = self.font()
        f.setPointSize(8)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())


class PillPanel(QFrame):
    """둥근 패널 (필 스타일)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("pill")
        self.setStyleSheet(
            f"""
            QFrame#pill {{
                background-color: {DX.BG_PANEL};
                border: 1px solid {DX.BG_PANEL_BORDER};
                border-radius: 20px;
            }}
            """
        )


class PerfMonitorDesign(QWidget):
    @staticmethod
    def _format_temp_html(prefix: str, temp: int) -> str:
        return (
            f"{prefix}:&nbsp;&nbsp;"
            f"<span style='color: {DX.CYAN_SOFT}; font-weight: 700;'>{temp}°C</span>"
        )

    @staticmethod
    def _format_cpu_temp_html(temp: int) -> str:
        return (
            "CPU:&nbsp;&nbsp;"
            f"<span style='color: {DX.GREEN_BRIGHT}; font-weight: 700;'>{temp}°C</span>"
        )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DEEPX — Performance Monitor")
        configure_overlay_window(self)
        # 콘텐츠에 맞춘 최소 높이(기존 320*0.7 대비 ~10% 이상 축소)
        self.setMinimumSize(360, 160)
        self.setStyleSheet(f"background-color: {DX.BG};")
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 4, 10, 4)
        outer.setSpacing(4)

        # 섹션 타이틀 스타일
        def section_title(text: str) -> QLabel:
            t = QLabel(text)
            t.setStyleSheet(
                f"color: {DX.TEXT}; font-size: 12px; font-weight: 700; "
                f"letter-spacing: 0.12em; background: transparent; border: none; "
                f"margin: 0; padding: 0;"
            )
            return t

        # --- CPU ---
        cpu_panel = PillPanel()
        cpu_lay = QVBoxLayout(cpu_panel)
        cpu_lay.setContentsMargins(12, 0, 12, 0)
        #cpu_lay.setSpacing(0)
        cpu_lay.addWidget(section_title("HOST CPU"))
        cpu_row = QHBoxLayout()
        cpu_row.setContentsMargins(0, 0, 0, 0)
        #cpu_row.setSpacing(0)
        self._cpu_label = QLabel("CPU: —")
        self._cpu_label.setStyleSheet(ROW_LABEL_STYLE)
        self._cpu_bar = GlowProgressBar(cpu_style=True)
        self._cpu_bar.setStyleSheet("background: transparent; border: none;")
        cpu_row.addWidget(self._cpu_label, 1)
        cpu_row.addWidget(self._cpu_bar, 2)
        cpu_row_w = QWidget()
        cpu_row_w.setStyleSheet("background: transparent;")
        cpu_row_w.setLayout(cpu_row)
        cpu_lay.addWidget(cpu_row_w)
        cpu_lay.addStretch(1)
        cpu_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        outer.addWidget(cpu_panel, 1)

        # --- NPU ---
        npu_panel = PillPanel()
        npu_lay = QVBoxLayout(npu_panel)
        npu_lay.setContentsMargins(12, 5, 12, 6)
        npu_lay.setSpacing(4)
        npu_lay.addWidget(section_title("NPU CORES"))

        self._npu_rows: List[Tuple[QLabel, GlowProgressBar]] = []
        for i in range(NPU_CORE_COUNT):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            lab = QLabel(f"NPU core {i}: —")
            lab.setStyleSheet(ROW_LABEL_STYLE)
            bar = GlowProgressBar()
            bar.setStyleSheet("background: transparent; border: none;")
            row.addWidget(lab, 1)
            row.addWidget(bar, 2)
            w = QWidget()
            w.setStyleSheet("background: transparent;")
            w.setLayout(row)
            npu_lay.addWidget(w)
            self._npu_rows.append((lab, bar))

        npu_lay.addStretch(1)
        npu_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        outer.addWidget(npu_panel, 2)

        psutil.cpu_percent(interval=None)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(UPDATE_INTERVAL_MS)
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._raise_overlay)
        self._overlay_timer.start(1500)
        self._refresh()

        self._drag_origin: Optional[QPoint] = None
        self.setFocusPolicy(Qt.StrongFocus)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)
        QShortcut(QKeySequence(Qt.Key_Q), self, activated=self.close)
        self.installEventFilter(self)
        for child in self.findChildren(QWidget):
            child.installEventFilter(self)

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        QTimer.singleShot(0, self._raise_overlay)
        QTimer.singleShot(250, self._raise_overlay)

    def _raise_overlay(self) -> None:
        if self.isVisible():
            self.show()
            self.raise_()

    def eventFilter(self, obj, event):  # noqa: ANN001
        if event.type() == QEvent.MouseButtonPress:
            me = event
            if isinstance(me, QMouseEvent) and me.button() == Qt.LeftButton:
                self._drag_origin = me.globalPos() - self.frameGeometry().topLeft()
        elif event.type() == QEvent.MouseButtonRelease:
            me = event
            if isinstance(me, QMouseEvent) and me.button() == Qt.LeftButton:
                self._drag_origin = None
        elif event.type() == QEvent.MouseMove:
            me = event
            if (
                isinstance(me, QMouseEvent)
                and (me.buttons() & Qt.LeftButton)
                and self._drag_origin is not None
            ):
                self.move(me.globalPos() - self._drag_origin)
        return False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin = event.globalPos() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            event.buttons() & Qt.LeftButton
            and self._drag_origin is not None
        ):
            self.move(event.globalPos() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(self.rect(), QColor(DX.BG))

        # 상단 시안 라디얼 (라이트 트레일 느낌)
        rg = QRadialGradient(w * 0.85, h * 0.08, min(w, h) * 0.45)
        rg.setColorAt(0.0, QColor(0, 120, 255, 38))
        rg.setColorAt(0.35, QColor(0, 200, 255, 14))
        rg.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, w, h, rg)

        lg = QLinearGradient(0, 0, w * 0.4, h * 0.3)
        lg.setColorAt(0.0, QColor(0, 100, 220, 25))
        lg.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(0, 0, w, h, lg)

    def _refresh(self) -> None:
        pct = psutil.cpu_percent(interval=None)
        v = int(round(min(100.0, max(0.0, pct))))
        self._cpu_bar.setValue(v)
        cpu_temp = try_host_cpu_temperature()
        if cpu_temp is not None:
            self._cpu_label.setText(self._format_cpu_temp_html(cpu_temp))
        else:
            self._cpu_label.setText("CPU: —")

        utils, ipc_err = fetch_npu_utilization(0)

        for i, (lab, bar) in enumerate(self._npu_rows):
            u = utils[i] if i < len(utils) else None
            if u is not None:
                bar.setValue(int(round(min(100.0, max(0.0, u)))))
            else:
                bar.setValue(0)

            temp = try_device_status_temperature(0, i)
            if u is not None and temp is not None:
                lab.setText(self._format_temp_html(f"NPU core {i}", temp))
            elif u is not None:
                lab.setText(f"NPU core {i}: ")
            elif temp is not None:
                lab.setText(
                    f"NPU core {i}: —%  ·  "
                    f"<span style='color: {DX.CYAN_SOFT}; font-weight: 700;'>{temp}°C</span>"
                )
            else:
                extra = f" ({ipc_err})" if ipc_err else ""
                lab.setText(f"NPU core {i}: —{extra}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = app.palette()
    pal.setColor(pal.Window, QColor(DX.BG))
    pal.setColor(pal.WindowText, QColor(DX.TEXT))
    app.setPalette(pal)

    w = PerfMonitorDesign()
    w.adjustSize()
    ag = QApplication.desktop().availableGeometry()
    margin = 5
    w.move(
        ag.x() + ag.width() - w.width() - margin,
        ag.y() + ag.height() - w.height() - margin,
    )
    w.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
