"""Integrate a source's shared receive radius and emission orientation.

Uses only actual positive/negative measurement coordinates. Radius is assumed
uniform1000..1500; type prior comes from leave-one-out population evidence; directional orientation is uniform.
This is a soft likelihood, never an absence or clearance certificate.
"""

import math
import numpy as np


def likelihood(belief, particles):
    prior = getattr(belief, "directional_prior", 0.5)
    P = np.asarray(particles, float)
    if not belief.positive:
        return np.ones(len(P))
    positive = np.asarray(belief.positive, float)
    d = positive[None, :, :] - P[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    lowr = np.maximum(1000.0, dist.max(axis=1))
    lowr = np.minimum(lowr, 1500.0)
    theta = np.arctan2(d[:, :, 1], d[:, :, 0])
    base = theta[:, :1]
    delta = (theta - base + math.pi) % (2 * math.pi) - math.pi
    lo = delta.max(axis=1) - math.pi / 2
    hi = delta.min(axis=1) + math.pi / 2
    if not belief.negative:
        omni = (1500 - lowr) / 500
        directional = omni * np.maximum(0, hi - lo) / (2 * math.pi)
        return (1 - prior) * omni + prior * directional
    negative = np.asarray(belief.negative, float)
    nd = negative[None, :, :] - P[:, None, :]
    nr = np.linalg.norm(nd, axis=2)
    nt = np.arctan2(nd[:, :, 1], nd[:, :, 0])
    anti = (nt - base + 2 * math.pi) % (2 * math.pi) - math.pi
    order = np.argsort(nr, axis=1)
    nr = np.take_along_axis(nr, order, axis=1)
    anti = np.take_along_axis(anti, order, axis=1)
    lows = np.maximum.accumulate(np.c_[lo, anti - math.pi / 2], axis=1)
    highs = np.minimum.accumulate(np.c_[hi, anti + math.pi / 2], axis=1)
    angular = np.maximum(0, highs - lows) / (2 * math.pi)
    breaks = np.c_[
        lowr, np.minimum(1500, np.maximum(lowr[:, None], nr)), np.full(len(P), 1500.0)
    ]
    directional = np.sum(np.diff(breaks, axis=1) * angular, axis=1) / 500
    omni = np.maximum(0, np.minimum(1500, nr[:, 0]) - lowr) / 500
    return (1 - prior) * omni + prior * directional
