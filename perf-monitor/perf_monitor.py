#!/usr/bin/env python3
"""
CPU + NPU 모니터 (PyQt5 + psutil).

NPU 코어 utilization은 dxtop 과 동일 소스(DX-RT GET_USAGE IPC)를 사용합니다.
ctypes 로 메인 프로세스에서 SysV 메시지 큐를 직접 호출하면 일부 환경에서
인터프리터 종료 시 세그폴트가 있어, `dxrt_ipc_query.py` 자식 프로세스로 조회합니다.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import psutil
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

NPU_CORE_COUNT = 3
UPDATE_INTERVAL_MS = 2000
_IPC_SCRIPT = Path(__file__).resolve().parent / "dxrt_ipc_query.py"
_IPC_TIMEOUT_SEC = 3.0


def configure_overlay_window(window: QWidget) -> None:
    """Apply stronger top-most flags for overlay-style monitor windows."""
    window.setWindowFlags(
        window.windowFlags()
        | Qt.WindowStaysOnTopHint
        | Qt.FramelessWindowHint
        | Qt.Tool
        | Qt.X11BypassWindowManagerHint
    )
    window.setAttribute(Qt.WA_ShowWithoutActivating, True)


def fetch_npu_utilization(device_id: int = 0) -> Tuple[List[Optional[float]], Optional[str]]:
    """
    코어 0..2 utilization (%). 실패 시 (전부 None, 에러 문자열).
    """
    if not _IPC_SCRIPT.is_file():
        return [None] * NPU_CORE_COUNT, f"missing {_IPC_SCRIPT.name}"

    try:
        cp = subprocess.run(
            [sys.executable, str(_IPC_SCRIPT), str(device_id)],
            capture_output=True,
            text=True,
            timeout=_IPC_TIMEOUT_SEC,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [None] * NPU_CORE_COUNT, "dxrt_ipc_query timeout"
    except OSError as e:
        return [None] * NPU_CORE_COUNT, str(e)

    line = (cp.stdout or "").strip().splitlines()[-1] if (cp.stdout or "").strip() else ""
    if not line.startswith("{"):
        err = (cp.stderr or "").strip() or line or f"exit {cp.returncode}"
        return [None] * NPU_CORE_COUNT, err

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return [None] * NPU_CORE_COUNT, f"bad json: {line[:80]}"

    util = payload.get("util")
    err = payload.get("error")
    if not isinstance(util, list) or len(util) < NPU_CORE_COUNT:
        return [None] * NPU_CORE_COUNT, str(err or "bad util[]")

    out: List[Optional[float]] = []
    for i in range(NPU_CORE_COUNT):
        v = util[i]
        out.append(float(v) if isinstance(v, (int, float)) else None)

    if all(x is None for x in out) and err:
        return out, str(err)
    return out, str(err) if err else None


def try_device_status_temperature(device_id: int, ch: int) -> Optional[int]:
    try:
        from dx_engine.device_status import DeviceStatus
    except ImportError:
        return None
    try:
        ds = DeviceStatus.get_current_status(device_id)
        return int(ds.get_temperature(ch))
    except Exception:
        return None


def _read_sysfs_temperature(path: Path) -> Optional[float]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    try:
        value = float(raw)
    except ValueError:
        return None

    # Linux thermal/hwmon normally exposes milli-Celsius. Accept direct Celsius too.
    if value > 1000.0:
        value /= 1000.0
    if value < -40.0 or value > 150.0:
        return None
    return value


def _try_rockchip_cpu_temperature() -> Optional[int]:
    thermal_root = Path("/sys/class/thermal")
    cpu_temps: List[float] = []
    soc_temps: List[float] = []

    for zone in sorted(thermal_root.glob("thermal_zone*")):
        try:
            zone_type = (zone / "type").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue

        temp = _read_sysfs_temperature(zone / "temp")
        if temp is None:
            continue

        if any(name in zone_type for name in ("bigcore", "littlecore", "cpu")):
            cpu_temps.append(temp)
        elif zone_type in {"soc-thermal", "soc_thermal"}:
            soc_temps.append(temp)

    if cpu_temps:
        return int(round(max(cpu_temps)))
    if soc_temps:
        return int(round(max(soc_temps)))
    return None


def _try_psutil_cpu_temperature() -> Optional[int]:
    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None

    selected: List[float] = []
    fallback: List[float] = []
    preferred_chips = ("coretemp", "k10temp", "cpu", "soc_thermal", "soc-thermal")
    preferred_labels = ("package", "core", "cpu", "tdie", "tctl")

    for chip_name, entries in (temps or {}).items():
        chip = chip_name.lower()
        for ent in entries:
            current = ent.current
            if current is None:
                continue
            label = (ent.label or "").lower()
            value = float(current)
            fallback.append(value)
            if any(name in chip for name in preferred_chips) or any(
                name in label for name in preferred_labels
            ):
                selected.append(value)

    if selected:
        return int(round(max(selected)))
    if fallback:
        return int(round(max(fallback)))
    return None


def try_host_cpu_temperature() -> Optional[int]:
    return _try_rockchip_cpu_temperature() or _try_psutil_cpu_temperature()


class PerfMonitor(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CPU / NPU monitor")
        configure_overlay_window(self)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)

        self._cpu_label = QLabel("CPU: —")
        self._cpu_bar = QProgressBar()
        self._cpu_bar.setRange(0, 100)
        self._cpu_bar.setFormat("%p%")
        root.addWidget(self._cpu_label)
        root.addWidget(self._cpu_bar)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        root.addWidget(QLabel("NPU cores"))

        self._npu_rows: List[Tuple[QLabel, QProgressBar]] = []
        for i in range(NPU_CORE_COUNT):
            row = QHBoxLayout()
            lab = QLabel(f"NPU core {i}: —")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setFormat("%p%")
            row.addWidget(lab, 1)
            row.addWidget(bar, 3)
            w = QWidget()
            w.setLayout(row)
            root.addWidget(w)
            self._npu_rows.append((lab, bar))

        psutil.cpu_percent(interval=None)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(UPDATE_INTERVAL_MS)
        self._overlay_timer = QTimer(self)
        self._overlay_timer.timeout.connect(self._raise_overlay)
        self._overlay_timer.start(1500)
        self._refresh()

    def showEvent(self, event) -> None:  # noqa: ANN001
        super().showEvent(event)
        QTimer.singleShot(0, self._raise_overlay)
        QTimer.singleShot(250, self._raise_overlay)

    def _raise_overlay(self) -> None:
        if self.isVisible():
            self.show()
            self.raise_()

    def _refresh(self) -> None:
        pct = psutil.cpu_percent(interval=None)
        v = int(round(min(100.0, max(0.0, pct))))
        self._cpu_bar.setValue(v)
        cpu_temp = try_host_cpu_temperature()
        if cpu_temp is not None:
            self._cpu_label.setText(f"CPU: {cpu_temp}°C")
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
                lab.setText(f"NPU core {i}: {temp}°C")
            elif u is not None:
                lab.setText(f"NPU core {i}: ")
            elif temp is not None:
                lab.setText(f"NPU core {i}: —%  ·  {temp}°C")
            else:
                extra = f" ({ipc_err})" if ipc_err else ""
                lab.setText(f"NPU core {i}: —{extra}")


def main() -> int:
    app = QApplication(sys.argv)
    window = PerfMonitor()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
