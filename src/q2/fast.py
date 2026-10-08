"""Optional compiled inner loop; portable Python fallback keeps identical model."""

import ctypes, subprocess, shutil, os
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
src = HERE / "fast_geometry.cpp"
libpath = HERE / "_fast_geometry.so"
fn = None
if shutil.which("clang++") and os.environ.get("Q2_PUREPY") != "1":
    try:
        if not libpath.exists() or libpath.stat().st_mtime < src.stat().st_mtime:
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
        fn = lib.wedge_diameter
        fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int] + [
            ctypes.c_double
        ] * 5
        fn.restype = ctypes.c_double
    except (OSError, subprocess.SubprocessError):
        fn = None


def prepare(p):
    return np.ascontiguousarray(p, dtype=np.float64).reshape(-1, 2)


def value(p, q, beta, alpha, radius):
    if fn:
        return fn(
            p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(p),
            float(q[0]),
            float(q[1]),
            beta,
            alpha,
            radius,
        )
    from geometry import wedge, outside, diameter

    if alpha <= 0:
        return 0.0
    points = list(map(tuple, p))
    clipped = wedge(points, q, beta, alpha) if alpha < np.pi / 2 else points
    return diameter(outside(clipped, q, radius))
