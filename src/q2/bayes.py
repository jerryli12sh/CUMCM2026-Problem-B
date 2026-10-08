"""Prior sampling and direct Monte Carlo reference integration."""

import math
import numpy as np
from scipy.stats import triang, truncnorm, uniform, beta, norm
from geometry import (
    Geometry,
    A,
    diameter,
    disk,
    wedge,
    outside,
    v,
    fast_prepare,
    fast_value,
)


def error_dist(kind):
    if kind == "uniform":
        return uniform(-A, 2 * A)
    if kind == "triangle":
        return triang(0.5, loc=-A, scale=2 * A)
    if kind == "biased":
        return beta(5, 2, loc=-A, scale=2 * A)
    sigma = float(kind.split("_")[1]) * A
    return truncnorm(-A / sigma, A / sigma, scale=sigma)


def samples(g, n, seed, kind="uniform", position="area", radius="uniform"):
    rng = np.random.default_rng(seed)
    dist = error_dist(kind)
    xs = []
    rs = []
    es = []
    count = 0
    while count < n:
        m = max(2048, 2 * (n - count))
        u = rng.random(m)
        r = np.sqrt(25 + (1500**2 - 25) * u) if position == "area" else 5 + 1495 * u
        e = dist.ppf(rng.random(m))
        phi = -e
        x = r[:, None] * np.column_stack((np.cos(phi), np.sin(phi)))
        if radius == "uniform":
            prob = (1500 - np.maximum(1000, r)) / 500
            rho = np.maximum(1000, r) + (1500 - np.maximum(1000, r)) * rng.random(m)
        else:
            rho = np.full(m, float(radius))
            prob = (r <= rho).astype(float)
        keep = g.valid_first(x) & (rng.random(m) < prob)
        xs.append(x[keep])
        rs.append(rho[keep])
        es.append(e[keep])
        count += int(keep.sum())
        if not keep.any() and len(xs) > 100:
            raise ValueError("First observation extremely unlikely under prior")
    return (
        np.concatenate(xs)[:n],
        np.concatenate(rs)[:n],
        np.concatenate(es)[:n],
        rng.standard_normal(n),
    )


def _table(g, q, bins=4096):
    po = disk(g.outer, q, 1500, True, 512)
    pi = disk(g.inner, q, 1500, False, 512)
    start, width = v.angle_range(po, q)
    step = width / bins
    lower = []
    upper = []
    pf = fast_prepare(po)
    pif = fast_prepare(pi)
    for i in range(bins):
        mid = start + (i + 0.5) * step
        lower.append(fast_value(pif, q, mid, max(0, A - step / 2), 5 + 1e-8))
        upper.append(fast_value(pf, q, mid, A + step / 2, 5 - 1e-8))
    return {
        "start": start,
        "width": width,
        "lower": np.array(lower),
        "upper": np.array(upper),
        "silent": [diameter(g.branch(q, 0, upper=False)), diameter(g.branch(q, 0))],
        "near": [diameter(g.branch(q, 1, upper=False)), diameter(g.branch(q, 1))],
    }


def losses(g, q, t, x, rho, e2):
    if math.hypot(*q) < 1e-12:
        return np.full(len(x), g.Dlower), np.full(len(x), g.D), np.full(len(x), 2)
    kind, b = g.feedback(x, rho, q, e2)
    lo = np.zeros(len(x))
    hi = lo.copy()
    for k, key in [(0, "silent"), (1, "near")]:
        lo[kind == k] = t[key][0]
        hi[kind == k] = t[key][1]
    idx = np.floor(
        ((b - t["start"]) % (2 * math.pi)) / t["width"] * len(t["lower"])
    ).astype(int)
    normal = kind == 2
    if np.any((idx[normal] < 0) | (idx[normal] >= len(t["lower"]))):
        raise AssertionError("bearing table missed normal feedback")
    lo[normal] = t["lower"][idx[normal]]
    hi[normal] = t["upper"][idx[normal]]
    return lo, hi, kind
