"""Conservative polygon envelopes and continuous coverage certificates."""

import itertools
import math
import numpy as np


def minimum_enclosing_circle(vertices, tol=1e-8):
    """Small-polygon support-circle enumeration, derived in Question 1."""
    if not vertices:
        raise ValueError("empty point set")
    candidates = [(p, 0.0) for p in vertices]
    for a, b in itertools.combinations(vertices, 2):
        center = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        candidates.append((center, math.dist(a, b) / 2))
    for a, b, c in itertools.combinations(vertices, 3):
        ux, uy = b[0] - a[0], b[1] - a[1]
        vx, vy = c[0] - a[0], c[1] - a[1]
        det = 2 * (ux * vy - uy * vx)
        if det == 0:
            continue
        uu = ux * ux + uy * uy
        vv = vx * vx + vy * vy
        center = (a[0] + (uu * vy - vv * uy) / det, a[1] + (ux * vv - vx * uu) / det)
        candidates.append((center, math.dist(center, a)))
    center, radius = min(
        (
            (c, r)
            for c, r in candidates
            if all(math.dist(c, p) <= r + tol for p in vertices)
        ),
        key=lambda x: x[1],
    )
    return {"center": center, "radius": radius}


ALPHA = math.radians(1.005)


def clip(poly, a, b, c):
    if not poly:
        return []
    c += 1e-8 * max(1, math.hypot(a, b))
    out = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        f = a * p[0] + b * p[1] - c
        g = a * q[0] + b * q[1] - c
        if (f <= 0) != (g <= 0):
            t = f / (f - g)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        if g <= 0:
            out.append(q)
    return out


def disk_polygon(center=(0, 0), radius=1800, n=96):
    r = radius / math.cos(math.pi / n)
    return [
        (
            center[0] + r * math.cos(2 * math.pi * i / n),
            center[1] + r * math.sin(2 * math.pi * i / n),
        )
        for i in range(n)
    ]


def intersect_disk(poly, q, radius, n=64):
    if not poly or max(math.dist(q, v) for v in poly) <= radius:
        return poly
    for i in range(n):
        t = 2 * math.pi * i / n
        a, b = math.cos(t), math.sin(t)
        poly = clip(poly, a, b, a * q[0] + b * q[1] + radius)
        if not poly:
            break
    return poly


def wedge(poly, q, beta):
    low, high = beta - ALPHA, beta + ALPHA
    a, b = math.sin(low), -math.cos(low)
    poly = clip(poly, a, b, a * q[0] + b * q[1])
    a, b = -math.sin(high), math.cos(high)
    return clip(poly, a, b, a * q[0] + b * q[1])


def hull(points):
    points = sorted(set(points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    upper = []
    for p in points:
        while len(lower) > 1 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) > 1 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def exclude_disk_hull(poly, q, radius):
    """Convex outer envelope; may retain holes but cannot remove a feasible point."""
    if not poly:
        return []
    radius = max(0, radius - 1e-7)
    out = [p for p in poly if math.dist(p, q) >= radius]
    for p, t in zip(poly, poly[1:] + poly[:1]):
        dx, dy = t[0] - p[0], t[1] - p[1]
        ux, uy = p[0] - q[0], p[1] - q[1]
        aa = dx * dx + dy * dy
        if aa < 1e-24:
            continue
        bb = 2 * (ux * dx + uy * dy)
        cc = ux * ux + uy * uy - radius * radius
        disc = bb * bb - 4 * aa * cc
        if disc < 0:
            continue
        for sign in (-1, 1):
            frac = (-bb + sign * math.sqrt(disc)) / (2 * aa)
            if 0 <= frac <= 1:
                out.append((p[0] + frac * dx, p[1] + frac * dy))
    return hull(out)


def contains(poly, p, tol=1e-5):
    if len(poly) == 1:
        return math.dist(poly[0], p) <= tol
    if len(poly) == 2:
        a, b = poly
        return abs(math.dist(a, p) + math.dist(p, b) - math.dist(a, b)) <= tol
    return bool(poly) and all(
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        >= -tol * max(1, math.dist(a, b))
        for a, b in zip(poly, poly[1:] + poly[:1])
    )


class Belief:
    def __init__(self, channel):
        self.channel = channel
        self.poly = disk_polygon()
        self.positive = []
        self.negative = []
        self.bearings = []
        self.failed = []
        self.cleared = False
        self.version = 0
        self._circle = None

    @property
    def known(self):
        return bool(self.positive)

    def circle(self):
        if self._circle is None:
            if not self.poly:
                raise ValueError(
                    f"Contradictory known-source envelope, channel {self.channel}"
                )
            result = minimum_enclosing_circle(self.poly)
            c = tuple(result["center"])
            self._circle = (c, max(math.dist(c, p) for p in self.poly) + 1e-7)
        return self._circle

    def _halfplane(self, negative, positive):
        a, b = negative[0] - positive[0], negative[1] - positive[1]
        c = (
            negative[0] ** 2 + negative[1] ** 2 - positive[0] ** 2 - positive[1] ** 2
        ) / 2
        self.poly = clip(self.poly, a, b, c)

    def observe(self, q, response):
        result = response["measure_result"]
        q = tuple(q)
        if result == "no_signal":
            self.negative.append(q)
            for p in self.positive:
                self._halfplane(q, p)
            if self.known:
                self.poly = exclude_disk_hull(self.poly, q, 1000)
        else:
            self.positive.append(q)
            if result == "direction":
                beta = math.radians(response["svd_deg"])
                self.bearings.append((q, beta))
                self.poly = wedge(self.poly, q, beta)
                self.poly = intersect_disk(self.poly, q, 1500)
            else:
                self.poly = intersect_disk(self.poly, q, 5)
            for n in self.negative:
                self._halfplane(n, q)
                self.poly = exclude_disk_hull(self.poly, n, 1000)
        self.version += 1
        self._circle = None
        if self.known and not self.poly:
            raise ValueError(f"Empty envelope for known channel {self.channel}")

    def clear_update(self, q, response):
        if response["clear_result"] == "success":
            self.cleared = True
        else:
            self.failed.append(tuple(q))
            self.poly = exclude_disk_hull(self.poly, q, 20)
            self._circle = None
            self.version += 1


def coverage_certificate(stations, limit=1000, min_cell=0.05):
    """Interval proof: every target-disk cell is inside a scanned reception disk.

    False means undecided or uncovered, never a false absence declaration.
    """
    stations = np.asarray(stations, dtype=float)
    if len(stations) == 0:
        return False, 0
    stack = [(0.0, 0.0, 1800.0)]
    checked = 0
    while stack:
        x, y, h = stack.pop()
        checked += 1
        if math.hypot(max(abs(x) - h, 0), max(abs(y) - h, 0)) > 1800:
            continue
        far = np.hypot(abs(stations[:, 0] - x) + h, abs(stations[:, 1] - y) + h)
        if float(far.min()) <= limit - 1e-6:
            continue
        if h * 2 <= min_cell:
            return False, checked
        nh = h / 2
        for dx in (-nh, nh):
            for dy in (-nh, nh):
                stack.append((x + dx, y + dy, nh))
    return True, checked


def farthest_uncovered(stations):
    """Find candidates from Voronoi vertices and target-circle boundary.

    Used only for action generation; absence is certified by interval coverage.
    """
    R = 1800.0
    candidates = [(R, 0.0), (-R, 0.0), (0.0, R), (0.0, -R)]
    for p in stations:
        r = math.hypot(*p)
        if r > 1e-9:
            candidates.append((-R * p[0] / r, -R * p[1] / r))
    for p, q in itertools.combinations(stations, 2):
        ax, ay = q[0] - p[0], q[1] - p[1]
        norm = math.hypot(ax, ay)
        if norm < 1e-8:
            continue
        c = (q[0] ** 2 + q[1] ** 2 - p[0] ** 2 - p[1] ** 2) / 2 / norm
        if abs(c) > R:
            continue
        nx, ny = ax / norm, ay / norm
        t = math.sqrt(max(0, R * R - c * c))
        candidates.extend(
            [(c * nx - t * ny, c * ny + t * nx), (c * nx + t * ny, c * ny - t * nx)]
        )
    for p, q, r in itertools.combinations(stations, 3):
        a, b = q[0] - p[0], q[1] - p[1]
        c, d = r[0] - p[0], r[1] - p[1]
        det = a * d - b * c
        if abs(det) < 1e-9:
            continue
        u = (q[0] ** 2 + q[1] ** 2 - p[0] ** 2 - p[1] ** 2) / 2
        v = (r[0] ** 2 + r[1] ** 2 - p[0] ** 2 - p[1] ** 2) / 2
        z = ((u * d - b * v) / det, (a * v - u * c) / det)
        if math.hypot(*z) <= R:
            candidates.append(z)
    pts = np.asarray(candidates)
    ss = np.asarray(stations)
    nearest = np.min(np.linalg.norm(pts[:, None, :] - ss[None, :, :], axis=2), axis=1)
    order = np.argsort(-nearest)
    return [(tuple(pts[i]), float(nearest[i])) for i in order[:30]]


def ring_stations(radius=1130.0, rotation=0.0):
    return [
        (
            radius * math.cos(rotation + k * math.pi / 3),
            radius * math.sin(rotation + k * math.pi / 3),
        )
        for k in range(6)
    ]
