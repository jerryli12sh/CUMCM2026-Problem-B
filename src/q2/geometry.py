"""General first observation; continuous station/bearing enclosures in float64.
The real-arithmetic inclusion argument is documented in the report. No claim of
outward-rounded interval certification. Coordinates here are local to first bearing.
"""

import sys, math, heapq
from pathlib import Path
import numpy as np
from fast import prepare as fast_prepare, value as fast_value
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
import q2_all_outcomes as v

clip = v.clip
wedge = v.posterior
outside = v.outside_vertices
A = math.pi / 180


def diameter(p):
    return float(pdist(np.asarray(p)).max()) if len(p) > 1 else 0.0


v.diameter = diameter


def disk(p, q, r, outer=True, n=256):
    if not p:
        return []
    # A whole polygon inside the disk (or its inscribed polygon's inball) needs no clipping.
    safe_radius = r if outer else r * math.cos(math.pi / n)
    if np.all(np.sum((np.asarray(p) - q) ** 2, axis=1) <= safe_radius**2):
        return p
    return v.cap_disk(p, q, r, outer, n)


def distance(p, q):
    if not p:
        return math.inf
    if v.inside(p, q):
        return 0.0
    p = np.asarray(p)
    z = np.roll(p, -1, axis=0) - p
    t = np.clip(
        np.sum((np.asarray(q) - p) * z, axis=1)
        / np.maximum(np.sum(z * z, axis=1), 1e-30),
        0,
        1,
    )
    return float(np.linalg.norm(p + t[:, None] * z - q, axis=1).min())


class Geometry:
    def __init__(self, S=(0, 0), theta=0, arc=48, disk_sides=1024):
        self.S = np.array(S, dtype=float)
        self.theta = float(theta)
        t = math.radians(theta)
        self.R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
        self.center = -self.R.T @ self.S
        po, pi = v.base.priors(arc)
        po = clip(po, -1, 0, -5 * math.cos(A))
        self.outer = disk(po, self.center, 1800, True, disk_sides)
        self.inner = disk(pi, self.center, 1800, False, disk_sides)
        if not self.outer:
            raise ValueError("Inconsistent first observation")
        self.D = diameter(self.outer)
        self.Dlower = diameter(self.inner)
        p = np.asarray(self.outer)
        self.bounds = [
            float(p[:, 0].min() - 1500),
            float(p[:, 1].min() - 1500),
            float(p[:, 0].max() + 1500),
            float(p[:, 1].max() + 1500),
        ]
        rmax = min(1499.99, max(np.linalg.norm(np.asarray(self.outer), axis=1)))
        rr = np.linspace(5.01, rmax, 101)
        aa = np.linspace(-A + 1e-10, A - 1e-10, 3)
        pts = np.array([(r * math.cos(a), r * math.sin(a)) for r in rr for a in aa])
        self.w = pts[self.valid_first(pts)]
        self.ii, self.jj = np.triu_indices(len(self.w), 1)
        self.pairdist = np.linalg.norm(self.w[self.ii] - self.w[self.jj], axis=1)

    def world(self, q):
        return (self.S + self.R @ np.asarray(q)).tolist()

    def valid_first(self, x):
        x = np.asarray(x)
        r = np.linalg.norm(x, axis=-1)
        return (
            (r > 5)
            & (r <= 1500 + 1e-9)
            & (np.abs(np.arctan2(x[..., 1], x[..., 0])) <= A + 1e-12)
            & (np.linalg.norm(x - self.center, axis=-1) <= 1800 + 1e-9)
        )

    def feedback(self, x, rho, q, e):
        d = np.linalg.norm(np.asarray(x) - q, axis=-1)
        beta = (
            np.arctan2(np.asarray(x)[..., 1] - q[1], np.asarray(x)[..., 0] - q[0]) + e
        )
        return np.where(d > rho, 0, np.where(d <= 5, 1, 2)), beta

    def contains(self, x, q, kind, beta=0):
        x = np.asarray(x)
        d = np.linalg.norm(x - q, axis=-1)
        r = np.linalg.norm(x, axis=-1)
        if kind == 0:
            ok = (d > 1000) & (d > r)
        elif kind == 1:
            ok = d <= 5 + 1e-8
        else:
            diff = np.arctan2(
                np.sin(np.arctan2(x[..., 1] - q[1], x[..., 0] - q[0]) - beta),
                np.cos(np.arctan2(x[..., 1] - q[1], x[..., 0] - q[0]) - beta),
            )
            ok = (d > 5) & (d <= 1500 + 1e-8) & (np.abs(diff) <= A + 1e-10)
        return self.valid_first(x) & ok

    def branch(self, q, kind, beta=0, upper=True, halfwidth=0, delta=0):
        q = np.asarray(q)
        p = self.outer if upper else self.inner
        sg = 1 if upper else -1
        if not p:
            return []
        if kind == 0:
            # h(g,q)=g.q-|q|²/2<0; perturbation <=delta*|g-q|+delta²/2.
            M = max(np.linalg.norm(np.asarray(p) - q, axis=1), default=0)
            extra = delta * M + delta * delta / 2 + 1e-8
            p = clip(p, *q, float(q @ q / 2 + sg * extra))
            return outside(p, q, 1000 - sg * (delta + 1e-8))
        if kind == 1:
            return disk(p, q, max(0, 5 + sg * delta), upper, 128)
        m = distance(self.outer, q)
        if delta and m <= delta:
            return p if upper else []
        eta = math.asin(min(1, delta / m)) if delta else 0
        alpha = A + sg * (halfwidth + eta)
        if alpha <= 0:
            return []
        p = disk(p, q, 1500 + sg * delta, upper, 256)
        p = wedge(p, q, beta, alpha) if alpha < math.pi / 2 else p
        return outside(p, q, max(0, 5 - sg * (delta + 1e-8)))

    def witness_lower(self, q, delta=0):
        q = np.asarray(q)
        d = np.linalg.norm(self.w - q, axis=1)
        r = np.linalg.norm(self.w, axis=1)
        silent = d - delta > np.maximum(1000, r)
        lo = diameter(self.w[silent])
        near = d + delta <= 5
        lo = max(lo, diameter(self.w[near]))
        eligible = (d - delta > 5) & (d + delta <= 1500)
        a = np.arctan2(self.w[:, 1] - q[1], self.w[:, 0] - q[0])
        eta = np.arcsin(np.minimum(1, delta / np.maximum(d, 1e-30)))
        i, j = self.ii, self.jj
        da = np.abs(np.arctan2(np.sin(a[i] - a[j]), np.cos(a[i] - a[j])))
        # Coupled angular difference derivative cancels common rotation: |grad(angle_i-angle_j)|=|Gi-Gj|/(di*dj).
        change = (
            delta
            * self.pairdist
            / (np.maximum(d[i] - delta, 1e-12) * np.maximum(d[j] - delta, 1e-12))
        )
        good = (
            eligible[i]
            & eligible[j]
            & (da + np.minimum(eta[i] + eta[j], change) <= 2 * A - 1e-12)
        )
        if good.any():
            lo = max(lo, float(self.pairdist[good].max()))
        return lo

    def bounds_at(
        self, q, delta=0, tol=0.1, max_nodes=512, cutoff=math.inf, angle_eta=None
    ):
        q = tuple(map(float, q))
        if delta == 0 and math.hypot(*q) < 1e-12:
            return {"lo": self.Dlower, "hi": self.D, "nodes": 0}
        sl = diameter(self.branch(q, 0, upper=False, delta=delta))
        su = diameter(self.branch(q, 0, delta=delta))
        nl = diameter(self.branch(q, 1, upper=False, delta=delta))
        nu = min(10, self.D)  # any actual near set is within its actual 5 m disk
        lower = max(sl, nl)
        if lower > cutoff:
            return {"lo": lower, "hi": self.D, "nodes": 0}
        if delta and distance(self.outer, q) <= delta:
            return {"lo": lower, "hi": self.D, "nodes": 0}
        po = disk(self.outer, q, 1500 + delta, True, 256)
        if not po:
            return {"lo": lower, "hi": su, "nodes": 0}
        start, width = v.angle_range(po, q)
        eta = math.asin(min(1, delta / distance(self.outer, q))) if delta else 0
        eta_lower = eta
        if angle_eta is not None:
            eta = min(eta, angle_eta)
            eta_lower = eta
        start -= eta
        width = min(2 * math.pi, width + 2 * eta)
        pi = disk(self.inner, q, 1500 - delta, False, 256)
        po_fast = fast_prepare(po)
        pi_fast = fast_prepare(pi)
        heap = []
        count = 0
        sample_upper_floor = max(su, nu)

        def add(a, b):
            nonlocal lower, count, sample_upper_floor
            mid = (a + b) / 2
            lo = fast_value(pi_fast, q, mid, A - eta_lower, 5 + delta + 1e-8)
            hi = fast_value(
                po_fast, q, mid, A + eta + (b - a) / 2, max(0, 5 - delta - 1e-8)
            )
            if delta:
                sample_upper_floor = max(
                    sample_upper_floor,
                    fast_value(po_fast, q, mid, A + eta, max(0, 5 - delta - 1e-8)),
                )
            lower = max(lower, lo)
            count += 1
            heapq.heappush(heap, (-hi, count, a, b))

        for i in range(12):
            add(start + width * i / 12, start + width * (i + 1) / 12)
        while heap and count < max_nodes:
            hi = max(su, nu, -heap[0][0])
            if hi - max(lower, sample_upper_floor) <= tol or lower > cutoff:
                break
            _, _, a, b = heapq.heappop(heap)
            mid = (a + b) / 2
            add(a, mid)
            add(mid, b)
        hi = min(self.D, max(su, nu, -heap[0][0] if heap else 0))
        if math.hypot(*q) <= delta:
            hi = self.D
        return {"lo": min(lower, hi), "hi": hi, "nodes": count, "silent": [sl, su]}

    def block_bounds(self, q, delta, tol=0.1, max_nodes=160, cutoff=math.inf):
        r = self.bounds_at(q, delta, tol, max_nodes, cutoff)
        m = distance(self.outer, q)
        if delta <= 0 or m <= delta or r["lo"] > cutoff:
            return r
        # Every indistinguishable pair has angular difference perturbation <=
        # delta*pair_distance/(m-delta)^2. Bootstrap with an already valid D bound.
        for _ in range(4):
            eta = delta * r["hi"] / (2 * (m - delta) ** 2)
            if eta >= math.asin(min(1, delta / m)):
                break
            z = self.bounds_at(q, delta, tol, max_nodes, cutoff, angle_eta=eta)
            improvement = r["hi"] - z["hi"]
            r["hi"] = min(r["hi"], z["hi"])
            r["lo"] = max(r["lo"], z["lo"])
            if improvement < 0.08:
                break
        return r
