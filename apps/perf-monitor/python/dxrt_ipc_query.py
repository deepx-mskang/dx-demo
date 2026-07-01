#!/usr/bin/env python3
"""
DX-RT GET_USAGE IPC (dxtop 과 동일). 요청은 #pragma pack(1) IPCClientMessage 바이너리를
struct 로만 조립해 ctypes 중첩 Structure 이슈를 피한다.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import struct
import sys
import time

IPC_CREAT = 0o1000
KEY_TO_SERVER = 0x2A020467
KEY_TO_CLIENT = 0x54020467
SERVER_MSG_MTYPE = 101
REQUEST_GET_USAGE = 17
RESPONSE_GET_USAGE_RESULT = 18
IPC_NOWAIT = 2048
N_CORE = 3
TIMEOUT_SEC = 2.0
POLL_SEC = 0.01

MSG_CLIENT = 164
MSG_SERVER = 96


class QueuedMsg(ctypes.Structure):
    _fields_ = [("mtype", ctypes.c_long), ("body", ctypes.c_char * 1024)]


def _mtext_ptr(msg: QueuedMsg) -> int:
    """SysV mtext 는 mtype(long) 직후부터; msg.body 에 직접 memmove 하면 일부 환경에서 깨짐."""
    return ctypes.addressof(msg) + ctypes.sizeof(ctypes.c_long)


def build_get_usage_request(device_id: int, core_id: int, pid: int) -> bytes:
    """IPCClientMessage, 1-byte aligned (DXRT)."""
    buf = bytearray(MSG_CLIENT)
    struct.pack_into("<IIQ", buf, 0, REQUEST_GET_USAGE, device_id, core_id)
    struct.pack_into("<i", buf, 16, pid)
    struct.pack_into("<q", buf, 20, pid)
    struct.pack_into("<i", buf, 28, 0)
    # 32 .. MSG_CLIENT-12 : npu_acc + padding 전부 0
    struct.pack_into("<i", buf, MSG_CLIENT - 12, -1)
    struct.pack_into("<Q", buf, MSG_CLIENT - 8, 0)
    return bytes(buf)


def _one_core(
    libc: ctypes.CDLL,
    q_srv: int,
    q_cli: int,
    pid: int,
    device_id: int,
    core_id: int,
) -> tuple[float | None, str | None]:
    payload = build_get_usage_request(device_id, core_id, pid)
    msg = QueuedMsg(mtype=SERVER_MSG_MTYPE)
    ctypes.memmove(_mtext_ptr(msg), payload, MSG_CLIENT)

    ctypes.set_errno(0)
    if libc.msgsnd(q_srv, ctypes.byref(msg), MSG_CLIENT, 0) != 0:
        e = ctypes.get_errno()
        return None, os.strerror(e) if e else "msgsnd 실패"

    deadline = time.monotonic() + TIMEOUT_SEC
    n = -1
    while time.monotonic() < deadline:
        ctypes.set_errno(0)
        n = libc.msgrcv(q_cli, ctypes.byref(msg), MSG_SERVER, pid, IPC_NOWAIT)
        if n >= 0:
            break
        err = ctypes.get_errno()
        if err in (0, errno.ENOMSG):
            time.sleep(POLL_SEC)
            continue
        return None, os.strerror(err)

    if n < 0:
        return None, "msgrcv timeout"

    raw = ctypes.string_at(_mtext_ptr(msg), MSG_SERVER)
    code, _dev_r, result, data_u64 = struct.unpack_from("<IIIQ", raw, 0)
    if result != 0 or code != RESPONSE_GET_USAGE_RESULT:
        return None, f"IPC result={result} code={code}"

    pct = min(100.0, max(0.0, data_u64 / 10.0))
    return pct, None


def main() -> None:
    if len(sys.argv) < 2:
        sys.stdout.write(json.dumps({"error": "usage: dxrt_ipc_query.py <device_id>", "util": [None] * N_CORE}) + "\n")
        sys.stdout.flush()
        os._exit(2)

    device_id = int(sys.argv[1])
    libc = ctypes.CDLL("libc.so.6")
    libc.msgsnd.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    libc.msgsnd.restype = ctypes.c_int
    libc.msgrcv.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_long,
        ctypes.c_int,
    ]
    libc.msgrcv.restype = ctypes.c_ssize_t

    pid = os.getpid()
    q_srv = libc.msgget(KEY_TO_SERVER, IPC_CREAT | 0o666)
    q_cli = libc.msgget(KEY_TO_CLIENT, IPC_CREAT | 0o666)
    if q_srv < 0 or q_cli < 0:
        sys.stdout.write(json.dumps({"error": "msgget failed", "util": [None] * N_CORE}) + "\n")
        sys.stdout.flush()
        os._exit(1)

    out: list[float | None] = []
    last_err: str | None = None
    for core in range(N_CORE):
        pct, err = _one_core(libc, q_srv, q_cli, pid, device_id, core)
        out.append(pct)
        if err:
            last_err = err

    if all(x is None for x in out):
        sys.stdout.write(json.dumps({"error": last_err or "unknown", "util": out}) + "\n")
    else:
        sys.stdout.write(json.dumps({"error": None, "util": out}) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
