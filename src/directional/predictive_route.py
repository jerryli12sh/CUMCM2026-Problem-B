"""Unseen-source prior scores expected FUTURE measurement time, not entropy.

Particles are a fixed quadrature independent of any live environment or seed.
They never provide an absence certificate. Hard completion is inherited.
"""

import math
import numpy as np
from route_optimizer import optimize_route


def survival_table(bot, scans):
    if bot.particle_cache is None:
        # Deterministic irrational rotations; no simulator sampling or internals.
        k = np.arange(1, 8193, dtype=float)
        r = 1770 * np.sqrt((k * 0.7548776662466927) % 1)
        a = 2 * np.pi * ((k * 0.5698402909980532) % 1)
        xy = np.column_stack((r * np.cos(a), r * np.sin(a)))
        ranges = 1000 + 500 * ((k * 0.438579021) % 1)
        angle = 2 * np.pi * ((k * 0.328173615) % 1)
        normals = np.column_stack((np.cos(angle), np.sin(angle)))
        bot.particle_cache = (xy, ranges, normals)
    xy, ranges, normals = bot.particle_cache

    def visible(q):
        delta = np.asarray(q) - xy
        om = (delta * delta).sum(axis=1) <= ranges * ranges
        dr = om & ((delta * normals).sum(axis=1) >= 0)
        return np.r_[om, dr]

    unseen = np.ones(2 * len(xy), dtype=bool)
    for q in bot.discovery_stations:
        unseen &= ~visible(q)
    return np.array([visible(q)[unseen] for q in scans]), unseen.mean()
