"""Observation-only route refinement; frozen v5 remains the comparison policy."""

import math
import numpy as np
from geometry import coverage_certificate
from robot import Robot


def open_route(points, scan_flags=None, discovery_weight=0.0):
    """Multi-start open route with reversal and one/two-node relocation.

    Node 0 is the fixed current position. A zero-cost sink makes endpoints free.
    A scan-delay term is an experimental proxy, not an information guarantee.
    """
    n = len(points) - 1
    if n <= 1:
        return list(range(1, n + 1))
    p = np.asarray(points, dtype=float)
    distances = np.zeros((n + 2, n + 2))
    distances[: n + 1, : n + 1] = np.linalg.norm(p[:, None] - p[None, :], axis=2)
    sink = n + 1
    flags = [False] + list(scan_flags or [False] * n) + [False]
    scan_count = max(1, sum(flags))

    def cost(route):
        elapsed = total = 0.0
        for a, b in zip(route, route[1:]):
            elapsed += distances[a, b]
            if flags[b]:
                total += discovery_weight * elapsed / scan_count
        return elapsed + total

    starts = []
    for first in range(1, n + 1):
        route = [0, first]
        left = set(range(1, n + 1)) - {first}
        while left:
            q = min(left, key=lambda i: distances[route[-1], i])
            route.append(q)
            left.remove(q)
        starts.append(route + [sink])
    # Insertion starts explore orderings not obtained by nearest-neighbour.
    for first in sorted(range(1, n + 1), key=lambda i: distances[0, i])[:3]:
        route = [0, first, sink]
        left = set(range(1, n + 1)) - {first}
        while left:
            _, q, j = min(
                (distances[a, q] + distances[q, b] - distances[a, b], q, j)
                for q in left
                for j, (a, b) in enumerate(zip(route, route[1:]), 1)
            )
            route.insert(j, q)
            left.remove(q)
        starts.append(route)

    best = None
    best_cost = math.inf
    seen = set()
    for route in starts:
        if tuple(route) in seen:
            continue
        seen.add(tuple(route))
        value = cost(route)
        for _ in range(12):
            improvement = 0.0
            replacement = None
            for i in range(1, n):
                for j in range(i + 1, n + 1):
                    delta = (
                        distances[route[i - 1], route[j]]
                        + distances[route[i], route[j + 1]]
                        - distances[route[i - 1], route[i]]
                        - distances[route[j], route[j + 1]]
                    )
                    if discovery_weight == 0 and delta >= improvement - 1e-7:
                        continue
                    trial = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                    change = cost(trial) - value if discovery_weight else delta
                    if change < improvement - 1e-7:
                        improvement, replacement = change, trial
            for width in (1, 2):
                for i in range(1, n - width + 2):
                    block = route[i : i + width]
                    a, b = route[i - 1], route[i + width]
                    removed = (
                        distances[a, b]
                        - distances[a, block[0]]
                        - distances[block[-1], b]
                    )
                    rest = route[:i] + route[i + width :]
                    for j in range(1, len(rest)):
                        if j == i:
                            continue
                        u, v = rest[j - 1], rest[j]
                        delta = (
                            removed
                            + distances[u, block[0]]
                            + distances[block[-1], v]
                            - distances[u, v]
                        )
                        if discovery_weight == 0 and delta >= improvement - 1e-7:
                            continue
                        trial = rest[:j] + block + rest[j:]
                        change = cost(trial) - value if discovery_weight else delta
                        if change < improvement - 1e-7:
                            improvement, replacement = change, trial
            if replacement is None:
                break
            route = replacement
            value = cost(route)
        if value < best_cost:
            best, best_cost = route, value
    return best[1:-1]


class RoutingRobot(Robot):
    def __init__(self, *args, discovery_weight=0.0, point_mode="mec", **kwargs):
        super().__init__(*args, **kwargs)
        self.discovery_weight = discovery_weight
        self.point_mode = point_mode
        self.point_cache = {}

    def source_target(self, channel):
        belief = self.beliefs[channel]
        center, radius = belief.circle()
        if self.point_mode == "mec" or radius < 19.95 or len(belief.poly) < 3:
            return center
        cached = self.point_cache.get(channel)
        if cached is not None and cached[0] == belief.version:
            return cached[1]
        # Deterministic area quadrature over the conservative polygon.
        # The radius-interval weights condition the SAME fixed R on all history.
        poly = np.asarray(belief.poly)
        a, bs, cs = poly[0], poly[1:-1], poly[2:]
        areas = (
            np.abs(
                (bs[:, 0] - a[0]) * (cs[:, 1] - a[1])
                - (bs[:, 1] - a[1]) * (cs[:, 0] - a[0])
            )
            / 2
        )
        if areas.sum() < 1e-12:
            return center
        k = np.arange(512, dtype=float) + 0.5
        triangles = np.searchsorted(np.cumsum(areas) / areas.sum(), k / len(k))
        u = np.sqrt((k * 0.6180339887498949) % 1)
        v = (k * 0.4142135623730950) % 1
        pts = (
            (1 - u[:, None]) * a
            + (u * (1 - v))[:, None] * bs[triangles]
            + (u * v)[:, None] * cs[triangles]
        )
        lower = np.full(len(k), 1000.0)
        upper = np.full(len(k), 1500.0)
        valid = np.linalg.norm(pts, axis=1) <= 1800
        for q in belief.positive:
            lower = np.maximum(lower, np.linalg.norm(pts - np.asarray(q), axis=1))
        for q in belief.negative:
            upper = np.minimum(upper, np.linalg.norm(pts - np.asarray(q), axis=1))
        for q in belief.failed:
            valid &= np.linalg.norm(pts - np.asarray(q), axis=1) > 20
        for q, beta in belief.bearings:
            delta = pts - np.asarray(q)
            angle = (np.arctan2(delta[:, 1], delta[:, 0]) - beta + math.pi) % (
                2 * math.pi
            ) - math.pi
            valid &= (np.abs(angle) <= math.radians(1.005)) & (
                np.linalg.norm(delta, axis=1) > 5
            )
        weights = np.maximum(upper - lower, 0) * valid
        if weights.sum() <= 1e-10:
            return center
        estimate = np.average(pts, axis=0, weights=weights)
        if self.point_mode == "median":
            for _ in range(40):
                w = weights / np.maximum(np.linalg.norm(pts - estimate, axis=1), 1e-6)
                updated = np.average(pts, axis=0, weights=w)
                if np.linalg.norm(updated - estimate) < 1e-4:
                    estimate = updated
                    break
                estimate = updated
        result = tuple(map(float, estimate))
        self.point_cache[channel] = (belief.version, result)
        return result

    def choose_joint_task(self):
        if self.check_coverage():
            self.pending_stations = []
        for q in list(self.pending_stations):
            others = [p for p in self.pending_stations if p != q]
            if coverage_certificate(self.discovery_stations + others)[0]:
                self.pending_stations.remove(q)
        tasks = [
            ("source", c, self.source_target(c))
            for c, b in self.beliefs.items()
            if b.known and not b.cleared
        ]
        tasks += [("scan", i, q) for i, q in enumerate(self.pending_stations)]
        if not tasks:
            return None
        order = open_route(
            [self.position] + [t[2] for t in tasks],
            [t[0] == "scan" for t in tasks],
            self.discovery_weight,
        )
        return tasks[order[0] - 1]

    def finish_source(self, ch):
        if self.point_mode == "mec":
            return super().finish_source(ch)
        belief = self.beliefs[ch]
        for attempt in range(10):
            if belief.cleared:
                return
            _, radius = belief.circle()
            if radius < 19.95:
                # Only the original conservative envelope certifies a clear.
                return super().finish_source(ch)
            center = self.source_target(ch)
            distance = math.dist(self.position, center)
            if radius <= self.try_radius and all(
                math.dist(center, f) > 3 for f in belief.failed
            ):
                if self.clear(center, ch):
                    return
                self.measure(self.position, ch)
                continue
            if distance > max(35, radius * 0.6):
                target = center
                if len(belief.bearings) == 1 and radius > 250:
                    theta = belief.bearings[0][1]
                    offset = min(90, radius * 0.12)
                    target = (
                        center[0] - offset * math.sin(theta),
                        center[1] + offset * math.cos(theta),
                    )
            else:
                _, p, q = max(
                    (math.dist(p, q), p, q) for p in belief.poly for q in belief.poly
                )
                theta = math.atan2(q[1] - p[1], q[0] - p[0])
                offset = max(25, min(90, radius * 0.65))
                target = (
                    center[0] - offset * math.sin(theta),
                    center[1] + offset * math.cos(theta),
                )
            if (ch, float(target[0]), float(target[1])) in self.measured:
                target = (target[0] + 17, target[1] + 23)
            self.measure(target, ch)
        self.fallback_source(ch)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from robot import HTTPTransport

    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:2027")
    p.add_argument("--robot-id", default="local-research")
    p.add_argument("--discovery-weight", type=float, default=0.0)
    p.add_argument("--point-mode", choices=["mec", "mean", "median"], default="mec")
    p.add_argument("--log", default="results/routing_actions.json")
    args = p.parse_args()
    robot = RoutingRobot(
        HTTPTransport(args.url),
        robot_id=args.robot_id,
        mode="joint_smart",
        discovery_weight=args.discovery_weight,
        point_mode=args.point_mode,
    )
    try:
        print(json.dumps(robot.run(), ensure_ascii=False))
    finally:
        Path(args.log).write_text(
            json.dumps(robot.log, ensure_ascii=False, indent=2) + "\n"
        )
