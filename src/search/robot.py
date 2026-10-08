"""Observation-only robot. Transport may be local or official HTTP; no truth API."""

import argparse
import json
import math
from pathlib import Path
import time
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen
import numpy as np
from geometry import Belief, coverage_certificate, farthest_uncovered, ring_stations


class HTTPTransport:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def __call__(self, path, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        request = Request(
            self.url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(4):
            try:
                with urlopen(request, timeout=8) as response:
                    return response.status, json.loads(response.read())
            except HTTPError as e:
                try:
                    body = json.loads(e.read())
                except ValueError:
                    body = {"accepted": False}
                return e.code, body
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise
                time.sleep(0.1 * (2**attempt))


class Robot:
    def __init__(
        self,
        transport,
        robot_id="local-research",
        mode="joint",
        try_radius=150,
        ring_radius=1130,
        audit=None,
        scan_known_radius=25,
        discovery_gain=150,
    ):
        self.transport = transport
        self.robot_id = robot_id
        self.mode = mode
        self.try_radius = try_radius
        self.ring_radius = ring_radius
        self.audit = audit
        self.scan_known_radius = scan_known_radius
        self.discovery_gain = discovery_gain
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_time = 0.0
        self.counter = 0
        self.beliefs = {c: Belief(c) for c in range(1, 21)}
        self.measured = set()
        self.discovery_stations = []
        self.covered = False
        self.log = []
        self.started = time.monotonic()
        self.remaining = 1200
        self.fallbacks = 0
        self.safe_clears = 0
        self.cover_checks = 0
        self.pending_stations = ring_stations(ring_radius)
        self.sweep_initialized = False
        pts = [
            (x, y)
            for x in range(-1800, 1801, 150)
            for y in range(-1800, 1801, 150)
            if x * x + y * y <= 1800**2
        ]
        pts += [
            (1800 * math.cos(t), 1800 * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 180, endpoint=False)
        ]
        self.discovery_probes = np.asarray(pts)

    def send(self, path, q=None, ch=None):
        self.counter += 1
        payload = {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": f"robot-{self.counter}",
        }
        if q is not None:
            payload.update(
                position={"x": float(q[0]), "y": float(q[1])}, channel=int(ch)
            )
        status, res = self.transport(path, payload)
        if status != 200 or res.get("accepted") is not True:
            raise RuntimeError(f"Action rejected: {path}, HTTP {status}, {res}")
        expected = self.virtual_time
        if q is not None:
            expected += round(math.dist(self.position, q) / 5 * 1e6) / 1e6
            if path == "/measure":
                expected += 5 + int(self.channel != ch)
            else:
                expected += 5 if res["clear_result"] == "success" else 3
            if abs(expected - res["virtual_time_s"]) > 2e-5:
                raise RuntimeError(
                    f'Timing mismatch: expected {expected}, got {res["virtual_time_s"]}'
                )
            self.position = tuple(map(float, q))
            if path == "/measure":
                self.channel = ch
        self.virtual_time = float(res["virtual_time_s"])
        self.log.append({"path": path, "request": payload, "response": res})
        return res

    def measure(self, q, ch):
        key = (ch, float(q[0]), float(q[1]))
        if key in self.measured:
            return None
        res = self.send("/measure", q, ch)
        self.measured.add(key)
        self.beliefs[ch].observe(q, res)
        if self.audit:
            self.audit(self, ch, "measure", False)
        if res["measure_result"] == "near":
            self.clear(q, ch, safe=True)
        return res

    def clear(self, q, ch, safe=False):
        if safe:
            self.safe_clears += 1
        res = self.send("/clear", q, ch)
        self.beliefs[ch].clear_update(q, res)
        if self.audit:
            self.audit(self, ch, "clear", safe)
        return res["clear_result"] == "success"

    def scan(self, q, all_known=False, force_discovery=True):
        improved = self.mode.startswith("joint") or self.mode == "event_v2"
        if improved and self.discovery_stations:
            self.check_coverage()
        discovery_needed = not self.covered
        if improved and not force_discovery and self.discovery_stations:
            if self.mode.startswith("joint_smart"):
                substitutes = False
                for old in self.pending_stations:
                    other = [p for p in self.pending_stations if p != old]
                    if coverage_certificate(
                        self.discovery_stations + [tuple(q)] + other
                    )[0]:
                        substitutes = True
                        break
                discovery_needed = discovery_needed and substitutes
            else:
                ss = np.asarray(self.discovery_stations)
                old = np.min(
                    np.linalg.norm(
                        self.discovery_probes[:, None, :] - ss[None, :, :], axis=2
                    ),
                    axis=1,
                )
                gain = int(
                    (
                        (old > 997)
                        & (
                            np.linalg.norm(
                                self.discovery_probes - np.asarray(q), axis=1
                            )
                            < 997
                        )
                    ).sum()
                )
                discovery_needed = discovery_needed and gain >= self.discovery_gain
        channels = []
        for c, b in self.beliefs.items():
            if b.cleared:
                continue
            if not b.known:
                if discovery_needed:
                    channels.append(c)
            elif all_known or b.circle()[1] > (
                max(self.try_radius, self.scan_known_radius)
                if improved
                else self.scan_known_radius
            ):
                center, radius = b.circle()
                # Skip a new bearing if its centre is too far away or its
                # measurement baseline is too short to justify another 5 s.
                if not improved or (
                    math.dist(center, q) < 1450
                    and all(math.dist(q, p) > 120 for p in b.positive)
                ):
                    channels.append(c)
        if self.channel in channels:
            channels.remove(self.channel)
            channels.insert(0, self.channel)
        for c in channels:
            self.measure(q, c)
        # Every still-undiscovered channel has been measured here, including any
        # previous visit; hence it shares this site's no-signal exclusion disk.
        if discovery_needed and not any(
            math.dist(q, p) < 1e-8 for p in self.discovery_stations
        ):
            self.discovery_stations.append(tuple(q))

    def check_coverage(self):
        if self.covered:
            return True
        if sum(b.known or b.cleared for b in self.beliefs.values()) == 16:
            self.covered = True
            return True
        self.cover_checks += 1
        self.covered, _ = coverage_certificate(self.discovery_stations)
        return self.covered

    def choose_source(self):
        channels = [c for c, b in self.beliefs.items() if b.known and not b.cleared]
        if not channels:
            return None
        centers = {c: self.beliefs[c].circle()[0] for c in channels}
        # Open nearest-neighbour route followed by 2-opt; no return to origin.
        order = []
        left = set(channels)
        pos = self.position
        while left:
            c = min(left, key=lambda i: math.dist(pos, centers[i]))
            order.append(c)
            left.remove(c)
            pos = centers[c]

        def length(route):
            pts = [self.position] + [centers[i] for i in route]
            return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))

        best = length(order)
        for _ in range(3):
            changed = False
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    trial = (
                        order[:i] + list(reversed(order[i : j + 1])) + order[j + 1 :]
                    )
                    value = length(trial)
                    if value < best - 1e-7:
                        order, best, changed = trial, value, True
            if not changed:
                break
        return order[0]

    def choose_joint_task(self):
        if self.check_coverage():
            self.pending_stations = []
        # Drop planned coverage stops only when the remaining real/planned scan
        # disks still have a continuous coverage certificate.
        for q in list(self.pending_stations):
            other = [p for p in self.pending_stations if p != q]
            ok, _ = coverage_certificate(self.discovery_stations + other)
            if ok:
                self.pending_stations.remove(q)
        tasks = [
            ("source", c, b.circle()[0])
            for c, b in self.beliefs.items()
            if b.known and not b.cleared
        ]
        tasks += [("scan", i, q) for i, q in enumerate(self.pending_stations)]
        if not tasks:
            return None

        def route_cost(order):
            pts = [self.position] + [tasks[i][2] for i in order]
            return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))

        if self.mode.startswith("joint_sweep"):
            scan_indices = [i for i, t in enumerate(tasks) if t[0] == "scan"]
            source_indices = [i for i, t in enumerate(tasks) if t[0] == "source"]
            scan_orders = [scan_indices]
            if not self.sweep_initialized and scan_indices:
                scan_orders = []
                for direction in [scan_indices, list(reversed(scan_indices))]:
                    for k in range(len(direction)):
                        scan_orders.append(direction[k:] + direction[:k])
            best_order = None
            best_cost = math.inf
            best_scans = None
            for scan_order in scan_orders:
                for sources in [source_indices, list(reversed(source_indices))]:
                    order = scan_order.copy()
                    for i in sources:
                        options = [
                            order[:j] + [i] + order[j:] for j in range(len(order) + 1)
                        ]
                        order = min(options, key=route_cost)
                    cost = route_cost(order)
                    for _ in range(3):
                        changed = False
                        for i in range(len(order) - 1):
                            for j in range(i + 1, len(order)):
                                if (
                                    sum(tasks[k][0] == "scan" for k in order[i : j + 1])
                                    > 1
                                ):
                                    continue
                                trial = (
                                    order[:i]
                                    + list(reversed(order[i : j + 1]))
                                    + order[j + 1 :]
                                )
                                value = route_cost(trial)
                                if value < cost - 1e-6:
                                    order, cost, changed = trial, value, True
                        if not changed:
                            break
                    if cost < best_cost:
                        best_order, best_cost, best_scans = order, cost, scan_order
            self.pending_stations = [tasks[i][2] for i in best_scans]
            self.sweep_initialized = True
            return tasks[best_order[0]]
        # Two initial route orientations reduce nearest-neighbour endpoint bias.
        best_order = None
        best_cost = math.inf
        seeds = sorted(
            range(len(tasks)), key=lambda i: math.dist(self.position, tasks[i][2])
        )[:3]
        for first in seeds:
            order = [first]
            left = set(range(len(tasks))) - {first}
            pos = tasks[first][2]
            while left:
                i = min(left, key=lambda i: math.dist(pos, tasks[i][2]))
                order.append(i)
                left.remove(i)
                pos = tasks[i][2]
            cost = route_cost(order)
            for _ in range(4):
                changed = False
                for i in range(len(order) - 1):
                    for j in range(i + 1, len(order)):
                        trial = (
                            order[:i]
                            + list(reversed(order[i : j + 1]))
                            + order[j + 1 :]
                        )
                        value = route_cost(trial)
                        if value < cost - 1e-6:
                            order, cost, changed = trial, value, True
                if not changed:
                    break
            if cost < best_cost:
                best_order, best_cost = order, cost
        first = tasks[best_order[0]]
        if self.mode.endswith("_flex") and first[0] == "scan":
            old = first[2]
            start = self.position
            end = tasks[best_order[1]][2] if len(best_order) > 1 else start
            dx, dy = end[0] - start[0], end[1] - start[1]
            norm = dx * dx + dy * dy
            t = (
                max(
                    0,
                    min(
                        1, ((old[0] - start[0]) * dx + (old[1] - start[1]) * dy) / norm
                    ),
                )
                if norm > 1e-12
                else 0
            )
            projection = (start[0] + t * dx, start[1] + t * dy)
            others = [q for q in self.pending_stations if q != old]
            for fraction in (1.0, 0.8, 0.6, 0.4, 0.2):
                q = (
                    old[0] + fraction * (projection[0] - old[0]),
                    old[1] + fraction * (projection[1] - old[1]),
                )
                if coverage_certificate(
                    self.discovery_stations + others + [q], limit=999
                )[0]:
                    self.pending_stations[self.pending_stations.index(old)] = q
                    first = ("scan", first[1], q)
                    break
        return first

    def finish_source(self, ch):
        b = self.beliefs[ch]
        for attempt in range(10):
            if b.cleared:
                return
            center, radius = b.circle()
            d = math.dist(self.position, center)
            if radius < 19.95:
                if d < 1e-9:
                    target = center
                else:
                    ux, uy = (self.position[0] - center[0]) / d, (
                        self.position[1] - center[1]
                    ) / d
                    limits = []
                    for p in b.poly:
                        vx, vy = center[0] - p[0], center[1] - p[1]
                        dot = vx * ux + vy * uy
                        limits.append(
                            -dot
                            + math.sqrt(
                                max(0, dot * dot + 19.95**2 - vx * vx - vy * vy)
                            )
                        )
                    shift = min(d, min(limits))
                    target = (center[0] + shift * ux, center[1] + shift * uy)
                if self.clear(target, ch, safe=True):
                    return
                raise RuntimeError("Certified clear failed")
            if radius <= self.try_radius and all(
                math.dist(center, f) > 3 for f in b.failed
            ):
                if self.clear(center, ch):
                    return
                self.measure(self.position, ch)
                continue
            # Approach a set centre; a modest transverse offset breaks the long
            # initial bearing geometry without making the large Q2-only detour.
            if d > max(35, radius * 0.6):
                target = center
                if len(b.bearings) == 1 and radius > 250:
                    theta = b.bearings[0][1]
                    offset = min(90, radius * 0.12)
                    target = (
                        center[0] - offset * math.sin(theta),
                        center[1] + offset * math.cos(theta),
                    )
            else:
                poly = b.poly
                pairs = [(math.dist(p, q), p, q) for p in poly for q in poly]
                _, p, q = max(pairs)
                theta = math.atan2(q[1] - p[1], q[0] - p[0])
                offset = max(25, min(90, radius * 0.65))
                target = (
                    center[0] - offset * math.sin(theta),
                    center[1] + offset * math.cos(theta),
                )
            key = (ch, float(target[0]), float(target[1]))
            if key in self.measured:
                target = (target[0] + 17, target[1] + 23)
            self.measure(target, ch)
        self.fallback_source(ch)

    def fallback_source(self, ch):
        """Finite continuous rectangle cover of the first positive bearing."""
        self.fallbacks += 1
        b = self.beliefs[ch]
        if not b.bearings:
            p = b.positive[0]
            if self.clear(p, ch, safe=True):
                return
            raise RuntimeError("near fallback failed")
        station, theta = b.bearings[0]
        c, s = math.cos(theta), math.sin(theta)
        width = 1500 * math.sin(math.radians(1.005))
        points = []
        for j in range(3):
            y = -width + (j + 0.5) * 2 * width / 3
            for i in (range(75) if j % 2 == 0 else reversed(range(75))):
                x = (i + 0.5) * 20
                points.append((station[0] + x * c - y * s, station[1] + x * s + y * c))
        # Visit in snake order. Do not discard cells via nonconservative sampling.
        for q in points:
            if self.clear(q, ch):
                return
        raise RuntimeError("Finite optical cover exhausted without success")

    def choose_discovery(self):
        if len(self.discovery_stations) >= 20:
            return next(
                q
                for q in ring_stations(self.ring_radius)
                if all(math.dist(q, p) > 1e-6 for p in self.discovery_stations)
            )
        ss = np.asarray(self.discovery_stations)
        distances = np.min(
            np.linalg.norm(self.discovery_probes[:, None, :] - ss[None, :, :], axis=2),
            axis=1,
        )
        uncovered = self.discovery_probes[distances > 997]
        far = farthest_uncovered(self.discovery_stations)
        candidates = ring_stations(self.ring_radius)
        candidates += [
            (r * math.cos(t), r * math.sin(t))
            for r in (1200, 1450, 1650)
            for t in np.linspace(0, 2 * math.pi, 36, endpoint=False)
        ]
        candidates += [(p[0] * 0.75, p[1] * 0.75) for p, d in far if d > 998]
        candidates = [
            q
            for q in candidates
            if all(math.dist(q, p) > 20 for p in self.discovery_stations)
        ]
        if len(uncovered) == 0:
            p, d = far[0]
            return (p[0] * 0.78, p[1] * 0.78)
        unknown = sum(not b.known and not b.cleared for b in self.beliefs.values())

        def score(q):
            gain = int((np.linalg.norm(uncovered - np.asarray(q), axis=1) <= 995).sum())
            cost = math.dist(q, self.position) / 5 + 6 * unknown
            return gain / max(1, cost)

        return max(candidates, key=score)

    def run(self, resume=False):
        if not resume:
            entered = self.send("/enter")
            self.started = time.monotonic()
            self.remaining = entered["remaining_real_duration_s"]
            self.scan((0.0, 0.0))
        if not resume and "oriented" in self.mode:
            self.orient_ring()
        if not resume and self.mode == "joint_adapt":
            detected = sum(b.known for b in self.beliefs.values())
            radius = (
                1650
                if detected == 0
                else (
                    1450
                    if detected <= 2
                    else 1250 if detected <= 4 else self.ring_radius
                )
            )
            self.pending_stations = ring_stations(radius)
        if not resume and self.mode == "scan_then_clear":
            for q in ring_stations(self.ring_radius):
                self.scan(q, all_known=True)
            if not self.check_coverage():
                raise RuntimeError("Fixed scan did not certify coverage")
        for _ in range(300):
            if time.monotonic() - self.started > self.remaining - 10:
                raise RuntimeError("Real-time budget almost exhausted")
            if self.mode.startswith("joint"):
                task = self.choose_joint_task()
                if task is None:
                    if not self.check_coverage():
                        raise RuntimeError("No pending tasks without coverage")
                    self.send("/exit")
                    return {
                        "certified_complete": True,
                        "discovery_station_count": len(self.discovery_stations),
                        "fallbacks": self.fallbacks,
                        "safe_clears": self.safe_clears,
                        "coverage_checks": self.cover_checks,
                    }
                kind, ch, q = task
                if kind == "scan":
                    self.scan(q)
                    self.pending_stations.remove(q)
                    continue
            else:
                ch = self.choose_source()
            if ch is not None:
                self.finish_source(ch)
                if self.mode != "scan_then_clear":
                    self.scan(
                        self.position,
                        force_discovery=not (
                            self.mode.startswith("joint") or self.mode == "event_v2"
                        ),
                    )
            elif self.check_coverage():
                self.send("/exit")
                return {
                    "certified_complete": True,
                    "discovery_station_count": len(self.discovery_stations),
                    "fallbacks": self.fallbacks,
                    "safe_clears": self.safe_clears,
                    "coverage_checks": self.cover_checks,
                }
            else:
                self.scan(self.choose_discovery())
        raise RuntimeError("Planning iteration limit reached")

    def orient_ring(self):
        centers = [
            b.circle()[0] for b in self.beliefs.values() if b.known and not b.cleared
        ]
        if not centers:
            return
        best = (math.inf, 0.0)
        for angle in np.linspace(0, math.pi / 3, 18, endpoint=False):
            points = (
                [self.position]
                + centers
                + ring_stations(self.ring_radius, float(angle))
            )
            distances = np.linalg.norm(
                np.asarray(points)[:, None, :] - np.asarray(points)[None, :, :], axis=2
            )
            left = set(range(1, len(points)))
            route = [0]
            while left:
                i = min(left, key=lambda i: distances[route[-1], i])
                route.append(i)
                left.remove(i)

            def cost(order):
                return sum(distances[a, b] for a, b in zip(order, order[1:]))

            value = cost(route)
            for _ in range(3):
                changed = False
                for i in range(1, len(route) - 1):
                    for j in range(i + 1, len(route)):
                        trial = (
                            route[:i]
                            + list(reversed(route[i : j + 1]))
                            + route[j + 1 :]
                        )
                        v = cost(trial)
                        if v < value - 1e-6:
                            route, value, changed = trial, v, True
                if not changed:
                    break
            if value < best[0]:
                best = (value, float(angle))
        self.pending_stations = ring_stations(self.ring_radius, best[1])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:2027")
    p.add_argument("--robot-id", default="local-research")
    p.add_argument("--mode", default="joint")
    p.add_argument("--log", default="robot_actions.json")
    args = p.parse_args()
    if args.mode == "joint_rollout":
        from lookahead import RolloutRobot

        robot_type = RolloutRobot
    else:
        robot_type = Robot
    robot = robot_type(HTTPTransport(args.url), robot_id=args.robot_id, mode=args.mode)
    try:
        result = robot.run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        Path(args.log).write_text(
            json.dumps(robot.log, ensure_ascii=False, indent=2) + "\n"
        )
