"""The v5 regional soft objective evaluated on an already selected task order."""

import math
import numpy as np
from predictive_route import survival_table
from rollout_type_prior import condition


class Cost:
    def __init__(self, bot):
        survival_table(bot, [])
        h = condition(bot)
        xy, radius, normals = bot.particle_cache
        n0 = len(xy)
        known = sum(b.known or b.cleared for b in bot.beliefs.values())
        eo = h["expected_omni"]
        ed = h["expected_directional"]
        ids = np.flatnonzero(h["alive"])
        oi = ids[ids < n0]
        di = ids[ids >= n0]
        oi = oi[:: max(1, len(oi) // 256)]
        di = di[:: max(1, len(di) // 256)]
        ids = np.r_[oi, di]
        self.ghost = xy[ids % n0]
        self.radii = radius[ids % n0]
        self.normals = normals[ids % n0]
        self.isdir = ids >= n0
        self.weights = np.r_[
            np.full(len(oi), eo / max(1, len(oi))),
            np.full(len(di), ed / max(1, len(di))),
        ]
        self.blank = 20 - known - h["expected"]
        self.keys = list(bot.route_keys)
        self.scan = np.array([kind == "scan" for kind, key in self.keys])
        self.points = np.array(
            [
                bot.source_target(key) if kind == "source" else key
                for kind, key in self.keys
            ],
            float,
        )
        self.start = np.array(bot.position)
        self.expected = h["expected"]

    def components(self, points=None):
        p = self.points if points is None else np.asarray(points, float)
        w = self.weights
        g = self.ghost
        edge = np.linalg.norm(np.diff(p, axis=0), axis=1) / 5
        travel = float(np.linalg.norm(self.start - p[0]) / 5 + edge.sum())
        if not len(w):
            return travel, 6 * self.blank * int(self.scan.sum()), 0.0, 0.0
        dg = np.linalg.norm(p[:, None, :] - g[None, :, :], axis=2) / 5
        delta = p[:, None, :] - g[None, :, :]
        vis = (np.sum(delta * delta, axis=2) <= self.radii**2) & (
            ~self.isdir | (np.sum(delta * self.normals, axis=2) >= 0)
        )
        vis[~self.scan] = False
        seen = np.logical_or.accumulate(vis, axis=0)
        if not np.all(seen[-1]):
            return math.inf, math.inf, math.inf, math.inf
        before = np.r_[np.zeros((1, len(w)), bool), seen[:-1]]
        fees = float(6 * np.sum(self.blank + ((~before)[self.scan] @ w)))
        insertion = np.r_[dg[:-1] + dg[1:] - edge[:, None], dg[-1:]]
        suffix = np.minimum.accumulate(insertion[::-1], axis=0)[::-1]
        first = np.argmax(vis, axis=0)
        future = float(w @ suffix[first, np.arange(len(w))])
        nodes = np.round(p / 150) * 150
        offgrid = np.linalg.norm(p - nodes, axis=1) > 1e-7
        mass = np.bincount(first, weights=w, minlength=len(p))
        node_distance = np.linalg.norm(p - nodes, axis=1) / 5
        penalty = np.r_[
            node_distance[:-1] + np.linalg.norm(nodes[:-1] - p[1:], axis=1) / 5 - edge,
            node_distance[-1:],
        ]
        anchor = float(
            6 * (w @ offgrid[first])
            + np.sum((1 - np.exp(-mass)) * np.maximum(0, penalty))
        )
        return travel, fees, future, anchor

    def first(self, point):
        p = self.points.copy()
        p[0] = point
        return sum(self.components(p))

    def first_occupancy(self):
        if not len(self.weights):
            return 0.0
        d = self.points[0] - self.ghost
        visible = (np.sum(d * d, axis=1) <= self.radii**2) & (
            ~self.isdir | (np.sum(d * self.normals, axis=1) >= 0)
        )
        return float(1 - np.exp(-(self.weights @ visible)))
