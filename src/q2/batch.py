"""Batch the unchanged wedge kernel; equivalent Python fallback."""

import ctypes, os, shutil, subprocess
from pathlib import Path
import numpy as np
from fast import value

HERE = Path(__file__).parent
src = HERE / "batch_geometry_v2.cpp"
libpath = HERE / "_batch_geometry_v2.so"
fn = None
if shutil.which("clang++") and os.environ.get("Q2_PUREPY") != "1":
    try:
        if not libpath.exists() or libpath.stat().st_mtime < max(
            src.stat().st_mtime, (HERE / "fast_geometry.cpp").stat().st_mtime
        ):
            subprocess.run(
                [
                    "clang++",
                    "-O3",
                    "-std=c++11",
                    "-shared",
                    "-fPIC",
                    str(src),
                    "-o",
                    str(libpath),
                ],
                check=True,
                capture_output=True,
            )
        lib = ctypes.CDLL(str(libpath))
        fn = lib.wedge_table
        ptr = ctypes.POINTER(ctypes.c_double)
        fn.argtypes = [ptr, ctypes.c_int] + [ctypes.c_double] * 4 + [ctypes.c_int, ptr]
        fn.restype = None
    except (OSError, subprocess.SubprocessError):
        pass


def values(p, q, alpha, radius, bins):
    out = np.empty(bins)
    if fn:
        fn(
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(p),
            float(q[0]),
            float(q[1]),
            alpha,
            radius,
            bins,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        for j in range(bins):
            out[j] = value(p, q, -np.pi + (j + 0.5) * 2 * np.pi / bins, alpha, radius)
    return out
