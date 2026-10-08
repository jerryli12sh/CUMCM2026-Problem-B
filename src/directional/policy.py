"""Directional beliefs, receipt-based discovery checks, joint tasks, and optical completion."""

import math
import time
from geometry import Belief, wedge, intersect_disk, clip
from short_baseline import LeanGradientRobot
from gradient import project_polygon
from coverage_geometry import rings, certify
from robot import Robot
from events import ReplanAfterBearing
from route_optimizer import optimize_route
from orientation_envelope import prune_orientation_envelope


def coverage_grid():
    return rings(8, 12, 999.0, 1866.0)


class DirectionalBelief(Belief):
    def __init__(self, channel):
        super().__init__(channel)
        self.orientation_diagnostics = []

    def observe(self, q, response):
        q = tuple(q)
        result = response["measure_result"]
        if result == "no_signal":
            self.negative.append(q)
        elif result in ("direction", "near"):
            self.positive.append(q)
            if result == "direction":
                beta = math.radians(response["svd_deg"])
                self.bearings.append((q, beta))
                self.poly = intersect_disk(wedge(self.poly, q, beta), q, 1500)
            else:
                self.poly = intersect_disk(self.poly, q, 5)
        else:
            raise RuntimeError("Unexpected measurement result: " + str(result))
        if self.positive and self.negative and len(self.poly) >= 3:
            self.poly, diagnostic = prune_orientation_envelope(
                self.poly, self.positive, self.negative
            )
            self.orientation_diagnostics.append(diagnostic)
        self.version += 1
        self._circle = None
        if self.known and not self.poly:
            raise RuntimeError("Empty conservative envelope")


class OfficialQ4Robot(LeanGradientRobot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beliefs = {c: DirectionalBelief(c) for c in range(1, 21)}
        self.required_stations = coverage_grid()
        self.geometry_certificate = certify(self.required_stations)
        if not self.geometry_certificate["certified"]:
            raise RuntimeError("Directional ring coverage not certified")
        self.pending_stations = [q for q in self.required_stations if q != (0.0, 0.0)]
        self.layout_selected = "rings21_directional"
        self.mirror_attempts = 0
        self.mirror_positive = 0
        self.route_keys = []
        self.route_diagnostics = []
        self.probe_anchor_diagnostics = []

    def measure(self, q, ch):
        # Obtain the ordinary response before the inherited event controller
        # can interrupt this step. Unknown-channel discovery is unchanged.
        finishing = self._finishing
        distance = math.dist(self.position, q)
        self._finishing = False
        try:
            response = super().measure(q, ch)
            b = self.beliefs[ch]
            if (
                response is not None
                and response["measure_result"] == "no_signal"
                and b.known
                and not b.cleared
                and b.bearings
                and q not in self.required_stations
            ):
                anchor, beta = b.bearings[-1]
                center, radius = b.circle()
                dx, dy = center[0] - anchor[0], center[1] - anchor[1]
                length = math.hypot(dx, dy)
                if length > 1:
                    ux, uy = dx / length, dy / length
                else:
                    ux, uy = math.cos(beta), math.sin(beta)
                vx, vy = q[0] - anchor[0], q[1] - anchor[1]
                side = -uy * vx + ux * vy
                if abs(side) >= 10:
                    candidates = [(q[0] + 2 * side * uy, q[1] - 2 * side * ux)]
                else:
                    offset = max(25.0, min(90.0, radius * 0.5))
                    candidates = [
                        (q[0] - offset * uy, q[1] + offset * ux),
                        (q[0] + offset * uy, q[1] - offset * ux),
                    ]
                for p in candidates:
                    if (ch, float(p[0]), float(p[1])) in self.measured:
                        continue
                    self.mirror_attempts += 1
                    recovered = Robot.measure(self, p, ch)
                    if recovered["measure_result"] != "no_signal":
                        self.mirror_positive += 1
                        break
        finally:
            self._finishing = finishing
        if finishing and response is not None and distance >= self.replan_distance:
            self.replans += 1
            raise ReplanAfterBearing()
        return response

    def check_coverage(self):
        self.cover_checks += 1
        # The problem states at most 16 sources with distinct channels.
        if sum(b.known or b.cleared for b in self.beliefs.values()) >= 16:
            self.covered = True
            return True
        # This depends on actually acknowledged measurements, never planned stops.
        self.covered = all(
            b.known
            or b.cleared
            or all(
                (c, float(q[0]), float(q[1])) in self.measured
                for q in self.required_stations
            )
            for c, b in self.beliefs.items()
        )
        return self.covered

    def scan(self, q, all_known=False, force_discovery=True):
        if not force_discovery:
            return
        prior = {c for c, b in self.beliefs.items() if b.known}
        channels = [c for c, b in self.beliefs.items() if not b.known and not b.cleared]
        if self.channel in channels:
            channels.remove(self.channel)
            channels.insert(0, self.channel)
        for c in channels:
            self.measure(q, c)
        if q not in self.discovery_stations:
            self.discovery_stations.append(tuple(q))
        # v26's small perpendicular probe remains only a speculative estimate.
        # v04: use the speculative 6 m estimate at every scan station.
        # Actual bearings still update the conservative envelope separately.
        for c, b in self.beliefs.items():
            if c in prior or not b.known or b.cleared or b.circle()[1] <= 100:
                continue
            anchor = q
            beta = next(
                beta
                for point, beta in reversed(b.bearings)
                if math.dist(point, q) < 1e-6
            )
            lattice = (float(150 * round(q[0] / 150)), float(150 * round(q[1] / 150)))
            if math.dist(lattice, q) > 1e-6:
                # Direct ordinary measurement prevents mirror recovery from
                # turning a cheap optional anchor into a long excursion.
                extra = Robot.measure(self, lattice, c)
                if b.cleared:
                    continue
                if extra is not None and extra["measure_result"] == "direction":
                    anchor = lattice
                    beta = math.radians(extra["svd_deg"])
                self.probe_anchor_diagnostics.append(
                    dict(
                        channel=c,
                        station=q,
                        anchor=anchor,
                        result=extra["measure_result"] if extra else "already_measured",
                    )
                )
            p = (anchor[0] - 6 * math.sin(beta), anchor[1] + 6 * math.cos(beta))
            r = self.measure(p, c)
            if r is None or r["measure_result"] != "direction" or b.cleared:
                continue
            delta = math.radians((r["svd_deg"] - math.degrees(beta) + 180) % 360 - 180)
            if delta >= -1e-12:
                continue
            distance = -6 / delta
            raw = (
                anchor[0] + distance * math.cos(beta),
                anchor[1] + distance * math.sin(beta),
            )
            if math.hypot(*raw) <= 1800:
                self.gradient_targets[c] = project_polygon(raw, b.poly)
                self.gradient_ranges[c] = distance
                self.gradient_diagnostics.append(
                    dict(channel=c, station=q, raw_estimate=raw)
                )

    def choose_joint_task(self):
        if self.check_coverage():
            self.pending_stations = []
        tasks = [
            ("source", c, self.source_target(c))
            for c, b in self.beliefs.items()
            if b.known and not b.cleared
        ]
        tasks += [("scan", i, q) for i, q in enumerate(self.pending_stations)]
        if not tasks:
            return None
        keys = [
            ("source", c) if kind == "source" else ("scan", q) for kind, c, q in tasks
        ]
        previous = [keys.index(key) for key in self.route_keys if key in keys]
        order, baseline, best = optimize_route(
            self.position, [t[2] for t in tasks], previous
        )
        self.route_keys = [keys[i] for i in order]
        self.route_diagnostics.append(
            dict(tasks=len(tasks), baseline_m=baseline, optimized_m=best)
        )
        return tasks[order[0]]

    def fallback_source(self, ch):
        # A frozen envelope is covered by closed 25 m squares. Their centres
        # are at most 25/sqrt(2) < 20 m from every feasible source position.
        self.fallbacks += 1
        b = self.beliefs[ch]
        poly = list(b.poly)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        points = []
        for i in range(math.floor(min(xs) / 25) - 1, math.floor(max(xs) / 25) + 2):
            for j in range(math.floor(min(ys) / 25) - 1, math.floor(max(ys) / 25) + 2):
                cell = poly
                for a, z, v in (
                    (-1, 0, -25 * i),
                    (1, 0, 25 * (i + 1)),
                    (0, -1, -25 * j),
                    (0, 1, 25 * (j + 1)),
                ):
                    cell = clip(cell, a, z, v)
                if cell:
                    points.append((25 * (i + 0.5), 25 * (j + 0.5)))
        while points:
            index = min(
                range(len(points)), key=lambda i: math.dist(self.position, points[i])
            )
            if self.clear(points.pop(index), ch):
                return
        raise RuntimeError("Finite optical envelope cover exhausted")

    def run(self, resume=False):
        if resume:
            raise ValueError("Resuming a case is disabled")
        entered = self.send("/enter")
        self.started = time.monotonic()
        self.remaining = entered["remaining_real_duration_s"]
        self.scan((0.0, 0.0))
        for _ in range(1000):
            if time.monotonic() - self.started > self.remaining - 10:
                raise RuntimeError("Official real-time allowance almost exhausted")
            task = self.choose_joint_task()
            if task is None:
                if not self.check_coverage() or any(
                    b.known and not b.cleared for b in self.beliefs.values()
                ):
                    raise RuntimeError("Missing discovery or clearance certificate")
                self.send("/exit")
                return dict(
                    certified_complete=True,
                    discovery_station_count=len(self.discovery_stations),
                    fallbacks=self.fallbacks,
                    gradient_successes=self.gradient_first_clear_successes,
                    coverage_kind="rings21_acknowledged_or_16_known",
                    geometry_certificate=self.geometry_certificate,
                    mirror_attempts=self.mirror_attempts,
                    mirror_positive=self.mirror_positive,
                    gradient_estimates=len(self.gradient_diagnostics),
                    gradient_guesses=self.gradient_guesses,
                    route_diagnostics=self.route_diagnostics,
                    probe_anchor_diagnostics=self.probe_anchor_diagnostics,
                    orientation_diagnostics={
                        str(c): b.orientation_diagnostics
                        for c, b in self.beliefs.items()
                        if b.orientation_diagnostics
                    },
                )
            kind, ch, q = task
            if kind == "scan":
                self.scan(q)
                self.pending_stations.remove(q)
            else:
                self.finish_source(ch)
        raise RuntimeError("Planning iteration budget exhausted")
