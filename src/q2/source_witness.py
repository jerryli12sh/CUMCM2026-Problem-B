"""Expectation lower bound from source-dependent, robust common-feedback witnesses.
This is NOT the old minimax witness: each distance is weighted by its compatible
normal-reading probability and then averaged over the true source posterior.
"""

import numpy as np
from geometry import A
import ctypes, subprocess, shutil, os
from pathlib import Path

_native = None
_src = Path(__file__).with_suffix(".cpp")
_lib = _src.with_name("_source_witness.so")
if shutil.which("clang++") and os.environ.get("Q2_PUREPY") != "1":
    try:
        if not _lib.exists() or _lib.stat().st_mtime < _src.stat().st_mtime:
            subprocess.run(
                [
                    "clang++",
                    "-O3",
                    "-std=c++11",
                    "-shared",
                    "-fPIC",
                    str(_src),
                    "-o",
                    str(_lib),
                ],
                check=True,
                capture_output=True,
            )
        _handle = ctypes.CDLL(str(_lib))
        _native = _handle.witness_lower
        _P = ctypes.POINTER(ctypes.c_double)
        _native.argtypes = [_P, ctypes.c_int, _P] + [ctypes.c_double] * 6 + [_P]
        _native.restype = None
    except (OSError, subprocess.SubprocessError):
        pass


def _lower(g, q, delta, x, silent_lower):
    r = np.linalg.norm(x, axis=1)
    unit = x / r[:, None]
    R = min(1499.99999, float(np.linalg.norm(g.outer, axis=1).max()) - 1e-5)
    rr = np.linspace(5.00001, R, 13)
    h = unit[:, None, :] * rr[None, :, None]
    valid = g.valid_first(h)
    z = x - q
    d = np.linalg.norm(z, axis=1)
    v = h - np.asarray(q)
    dh = np.linalg.norm(v, axis=2)
    eligible = (
        valid & (dh - delta > 5) & (dh + delta <= 1500) & ((d - delta > 5)[:, None])
    )
    sep = abs(r[:, None] - rr)
    dot = np.sum(z[:, None, :] * v, axis=2)
    cross = z[:, None, 0] * v[:, :, 1] - z[:, None, 1] * v[:, :, 0]
    angle = np.abs(np.arctan2(cross, dot))
    perturb = (
        delta
        * sep
        / (np.maximum(d - delta, 1e-12)[:, None] * np.maximum(dh - delta, 1e-12))
    )
    # Intersection of e in [-A,A] with a robust compatible-reading interval.
    pe = np.maximum(0, 1 - (angle + 2 * perturb) / (2 * A)) * eligible
    normal = np.max(sep * pe, axis=1)
    a = np.maximum(1000, r)
    den = np.maximum(1500 - a, 1e-12)
    psmin = np.clip((np.maximum(0, d - delta) - a) / den, 0, 1)
    psmax = np.clip((d + delta - a) / den, 0, 1)
    return normal * (1 - psmax) + silent_lower * psmin


def lower(g, q, delta, x, silent_lower):
    if _native:
        R = min(1499.99999, float(np.linalg.norm(g.outer, axis=1).max()) - 1e-5)
        rr = np.linspace(5.00001, R, 13)
        x = np.ascontiguousarray(x)
        out = np.empty(len(x))
        ptr = lambda z: z.ctypes.data_as(_P)
        _native(
            ptr(x),
            len(x),
            ptr(rr),
            float(q[0]),
            float(q[1]),
            float(delta),
            float(g.center[0]),
            float(g.center[1]),
            float(silent_lower),
            ptr(out),
        )
        return out
    if len(x) <= 8192:
        return _lower(g, q, delta, x, silent_lower)
    return np.concatenate(
        [
            _lower(g, q, delta, x[i : i + 8192], silent_lower)
            for i in range(0, len(x), 8192)
        ]
    )
