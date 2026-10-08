"""Allocation-free equivalent of numpy envelopes + mean/sample variance."""

import ctypes, subprocess, shutil, os, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
src = HERE / "moments.cpp"
libpath = HERE / "_moments.so"
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
        fn = lib.expected_moments
        P = ctypes.POINTER(ctypes.c_double)
        fn.argtypes = (
            [P, ctypes.c_int]
            + [ctypes.c_double] * 3
            + [P] * 4
            + [ctypes.c_int]
            + [ctypes.c_double] * 6
            + [P]
        )
        fn.restype = None
    except (OSError, subprocess.SubprocessError):
        pass


def stats(g, q, t, x, family, alpha=0.025):
    if fn:
        l = np.ascontiguousarray(t["lower"])
        h = np.ascontiguousarray(t["upper"])
        pl = np.r_[0, np.cumsum(l)] * t["step"]
        ph = np.r_[0, np.cumsum(h)] * t["step"]
        o = np.empty(4)
        ptr = lambda z: z.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        fn(
            ptr(x),
            len(x),
            float(q[0]),
            float(q[1]),
            t["delta"],
            ptr(l),
            ptr(h),
            ptr(pl),
            ptr(ph),
            len(l),
            *t["silent"],
            *t["near"],
            g.D,
            g.Dlower,
            ptr(o)
        )
        ml, mh, vl, vh = map(float, o)
    else:
        from expectation import envelopes

        l, h = envelopes(g, q, t, x)
        ml, mh = float(l.mean()), float(h.mean())
        vl, vh = float(l.var(ddof=1)), float(h.var(ddof=1))
    n = len(x)
    log = math.log(4 * family / alpha)
    B = t["bound"]
    el = math.sqrt(2 * max(0, vl) * log / n) + 7 * B * log / (3 * (n - 1))
    eh = math.sqrt(2 * max(0, vh) * log / n) + 7 * B * log / (3 * (n - 1))
    return {
        "lo": max(0, ml - el),
        "hi": min(B, mh + eh),
        "geom_lo": ml,
        "geom_hi": mh,
        "error_lo": el,
        "error_hi": eh,
        "n": n,
    }
