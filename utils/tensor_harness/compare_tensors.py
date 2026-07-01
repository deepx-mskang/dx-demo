#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os

def compare_files(file1: str, file2: str, atol: float = 1e-6):
    if not os.path.exists(file1):
        print(f"Error: {file1} does not exist.")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"Error: {file2} does not exist.")
        sys.exit(1)

    t1 = np.fromfile(file1, dtype=np.float32)
    t2 = np.fromfile(file2, dtype=np.float32)

    if t1.size != t2.size:
        print(f"FAIL: Size mismatch! {file1} has {t1.size} elements, {file2} has {t2.size} elements.")
        sys.exit(1)

    max_diff = np.max(np.abs(t1 - t2))
    mean_diff = np.mean(np.abs(t1 - t2))
    is_close = np.allclose(t1, t2, atol=atol)

    print("=== Tensor Comparison Results ===")
    print(f"File 1: {file1} (Size: {t1.size})")
    print(f"File 2: {file2} (Size: {t2.size})")
    print(f"Max Absolute Diff : {max_diff:.8e}")
    print(f"Mean Absolute Diff: {mean_diff:.8e}")
    
    if is_close:
        print(f"\n✅ PASS: Tensors are mathematically identical (Tolerance: {atol})")
    else:
        print(f"\n❌ FAIL: Tensors differ by more than tolerance {atol}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two float32 binary tensor dumps.")
    parser.add_argument("file1", help="Path to first .bin file (e.g. from Python)")
    parser.add_argument("file2", help="Path to second .bin file (e.g. from C++)")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for np.allclose")
    
    args = parser.parse_args()
    compare_files(args.file1, args.file2, args.atol)
