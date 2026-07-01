import numpy as np
import os

def dump_tensor(tensor: np.ndarray, filename: str):
    """
    Numpy 배열을 C++과 동일한 바이너리 포맷(.bin)으로 덤프합니다.
    """
    try:
        # C++ float32와 호환되도록 강제 캐스팅 후 1D로 변환하여 저장
        tensor.astype(np.float32).flatten().tofile(filename)
        print(f"[Harness] Dumped {tensor.size} floats to {filename}")
    except Exception as e:
        print(f"[Harness Error] Failed to dump tensor: {e}")
