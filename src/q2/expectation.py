"""Expected hard-feasible diameter. Geometry is inherited unchanged.
Uniform second-error/radius integrated analytically (Rao-Blackwellization).
Source samples remain iid from the declared first-observation posterior.
"""

import math
import numpy as np
from geometry import Geometry, A, disk, diameter, distance, fast_prepare, fast_value
from bayes import samples as old_samples, error_dist
from batch import values

MODELS = ("taper", "flat")


def samples(
    g, n, seed, model="taper", error="uniform", position="area", radius="uniform"
):
    if model == "taper":
        return old_samples(g, n, seed, error, position, radius)
    if model != "flat":
        raise ValueError(model)
    if error != "uniform" or position != "area" or radius != "uniform":
        raise ValueError("flat is the isolated posterior-position control")
    rng = np.random.default_rng(seed)
    xx = []
    rr = []
    ee = []
    count = 0
    while count < n:
        m = max(2048, 2 * (n - count))
        r = np.sqrt(25 + (1500**2 - 25) * rng.random(m))
        e = rng.uniform(-A, A, m)
        x = r[:, None] * np.column_stack((np.cos(e), -np.sin(e)))
        keep = g.valid_first(x)
        x = x[keep]
        r = r[keep]
        e = e[keep]
        a = np.maximum(1000, r)
        rho = a + (1500 - a) * rng.random(len(r))
        xx.append(x)
        rr.append(rho)
        ee.append(e)
        count += len(r)
    return (
        np.concatenate(xx)[:n],
        np.concatenate(rr)[:n],
        np.concatenate(ee)[:n],
        rng.standard_normal(n),
    )


def table(g, q, delta=0.0, bins=2048):
    """Periodic table. Includes both generated-bearing and feasible-source shifts.
    Whole-circle grid avoids missing support endpoints; bins are interval bounds.
    """
    q = np.asarray(q, float)
    m = distance(g.outer, q)
    step = 2 * math.pi / bins
    silent = [
        diameter(g.branch(q, 0, upper=False, delta=delta)),
        diameter(g.branch(q, 0, delta=delta)),
    ]
    near = [
        diameter(g.branch(q, 1, upper=False, delta=delta)),
        diameter(g.branch(q, 1, delta=delta)),
    ]
    if delta and m <= delta:
        lo = np.zeros(bins)
        hi = np.full(bins, g.D)
    else:
        eta = math.asin(min(1, delta / m)) if delta else 0.0
        # One eta for the generated reading and one for candidate-source direction.
        extra = 2 * eta + step / 2
        po = disk(g.outer, q, 1500 + delta, True, 512)
        pi = disk(g.inner, q, max(0, 1500 - delta), False, 512)
        pf = fast_prepare(po)
        pif = fast_prepare(pi)
        lo = values(pif, q, max(0, A - extra), 5 + delta + 1e-8, bins)
        hi = (
            values(pf, q, A + extra, max(0, 5 - delta - 1e-8), bins)
            if A + extra < math.pi / 2
            else np.full(bins, diameter(po))
        )
        # Relative-angle gradient cancels common rotation of the true and candidate source.
        # Current expanded set contains both, so its diameter bounds their separation.
        # Repeated tightening remains an enclosure, not a Lipschitz guess for M.
        if delta and m > delta:
            for _ in range(3):
                B = float(np.max(hi))
                tight = min(extra - step / 2, delta * B / (m - delta) ** 2) + step / 2
                if tight >= extra * 0.98:
                    break
                extra = tight
                lo = values(pif, q, max(0, A - extra), 5 + delta + 1e-8, bins)
                hi = (
                    values(pf, q, A + extra, max(0, 5 - delta - 1e-8), bins)
                    if A + extra < math.pi / 2
                    else np.full(bins, diameter(po))
                )
    return {
        "lower": lo,
        "upper": hi,
        "silent": silent,
        "near": near,
        "step": step,
        "delta": delta,
        "bound": min(g.D, max(float(np.max(hi)), silent[1], near[1])),
    }


def periodic_integral(arr, x):
    """Integral from -pi to x of periodic, piecewise constant arr."""
    n = len(arr)
    step = 2 * math.pi / n
    y = (x + math.pi) / step
    k = np.floor(y).astype(np.int64)
    f = y - k
    pref = np.r_[0, np.cumsum(arr)] * step
    return (k // n) * pref[-1] + pref[k % n] + f * step * arr[k % n]


def angular_mean(arr, b):
    return np.clip(
        (periodic_integral(arr, b + A) - periodic_integral(arr, b - A)) / (2 * A),
        0,
        np.max(arr),
    )


def envelopes(g, q, t, x):
    """For each true source, bound E_{rho,e2}[loss] uniformly over station block.
    Uniform rho|G and independent uniform e2 only. Bound handles feedback switches.
    """
    q = np.asarray(q, float)
    delta = t["delta"]
    r = np.linalg.norm(x, axis=1)
    z = x - q
    d = np.linalg.norm(z, axis=1)
    b = np.arctan2(z[:, 1], z[:, 0])
    dn = np.maximum(0, d - delta)
    dx = d + delta
    a = np.maximum(1000, r)
    den = np.maximum(1500 - a, 1e-12)
    pl = np.clip((dn - a) / den, 0, 1)
    ph = np.clip((dx - a) / den, 0, 1)
    nl = angular_mean(t["lower"], b)
    nh = angular_mean(t["upper"], b)
    sl, sh = t["silent"]
    kl, kh = t["near"]
    lo = np.minimum(pl * sl + (1 - pl) * nl, ph * sl + (1 - ph) * nl)
    hi = np.maximum(pl * sh + (1 - pl) * nh, ph * sh + (1 - ph) * nh)
    always = dx <= 5
    maybe = (dn <= 5) & ~always
    lo[always] = kl
    hi[always] = kh
    lo[maybe] = np.minimum(lo[maybe], kl)
    hi[maybe] = np.maximum(hi[maybe], kh)
    # Same-location fixed error is an exceptional action: no new information.
    if np.linalg.norm(q) <= delta:
        hi[:] = g.D
        if delta == 0:
            lo[:] = g.Dlower
    return np.maximum(0, lo), np.minimum(g.D, hi)


def confidence(lo, hi, bound, family=1, alpha=0.05):
    """Two-sided empirical Bernstein via sample variance; union over both envelopes.
    Maurer-Pontil bounded iid inequality; all arrays lie in [0,bound].
    """
    n = len(lo)
    log = math.log(4 * family / alpha)

    def err(v):
        return math.sqrt(2 * float(np.var(v, ddof=1)) * log / n) + 7 * bound * log / (
            3 * (n - 1)
        )

    el, eh = err(lo), err(hi)
    return {
        "lo": max(0, float(lo.mean()) - el),
        "hi": min(bound, float(hi.mean()) + eh),
        "geom_lo": float(lo.mean()),
        "geom_hi": float(hi.mean()),
        "error_lo": el,
        "error_hi": eh,
        "n": n,
    }


def point(g, q, x, bins=16384, family=1):
    t = table(g, q, bins=bins)
    lo, hi = envelopes(g, q, t, x)
    return {
        **confidence(lo, hi, t["bound"], family, alpha=0.025),
        "q": list(map(float, q)),
        "world": g.world(q),
        "bins": bins,
    }
