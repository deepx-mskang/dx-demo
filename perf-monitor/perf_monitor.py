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
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import psutil
from PyQt5.QtCore import QTimer
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


def _is_intel_x86_host() -> bool:
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64", "i386", "i686"}:
        return False

    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            cpuinfo = f.read().lower()
    except OSError:
        return False
    return "vendor_id" in cpuinfo and "genuineintel" in cpuinfo


def try_host_cpu_temperature() -> Optional[int]:
    if not _is_intel_x86_host():
        return None

    try:
        temps = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return None

    if not temps:
        return None

    package_temps: List[float] = []
    core_temps: List[float] = []

    for chip_name, entries in temps.items():
        chip = chip_name.lower()
        if "coretemp" not in chip:
            continue
        for ent in entries:
            current = ent.current
            if current is None:
                continue
            label = (ent.label or "").lower()
            if "package" in label:
                package_temps.append(float(current))
            else:
                core_temps.append(float(current))

    if package_temps:
        return int(round(max(package_temps)))
    if core_temps:
        return int(round(max(core_temps)))
    return None


class PerfMonitor(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CPU / NPU monitor")
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
        self._refresh()

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
