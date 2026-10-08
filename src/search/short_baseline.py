"""Perpendicular short-baseline localization, certified scan layouts, and ray recovery."""

import math
import numpy as np
from geometry import coverage_certificate
from routing import open_route
from selective_scan import SelectiveScanRobot
from gradient import project_polygon
from station_probes import AllVertexGradientRobot

GRID_LAYOUTS = {
    "rot0": [
        (1200.0, 0.0),
        (600.0, 1050.0),
        (-600.0, 1050.0),
        (-1200.0, 0.0),
        (-600.0, -1050.0),
        (600.0, -1050.0),
    ],
    "rot30": [
        (1050.0, 600.0),
        (0.0, 1200.0),
        (-1050.0, 600.0),
        (-1050.0, -600.0),
        (0.0, -1200.0),
        (1050.0, -600.0),
    ],
    "rot15": [
        (1200.0, 300.0),
        (300.0, 1200.0),
        (-900.0, 900.0),
        (-1200.0, -300.0),
        (-300.0, -1200.0),
        (900.0, -900.0),
    ],
    "rot45": [
        (900.0, 900.0),
        (-300.0, 1200.0),
        (-1200.0, 300.0),
        (-900.0, -900.0),
        (300.0, -1200.0),
        (1200.0, -300.0),
    ],
    "rot0_1350": [
        (1350.0, 0.0),
        (600.0, 1200.0),
        (-600.0, 1200.0),
        (-1350.0, 0.0),
        (-600.0, -1200.0),
        (600.0, -1200.0),
    ],
    "rot30_1350": [
        (1200.0, 600.0),
        (0.0, 1350.0),
        (-1200.0, 600.0),
        (-1200.0, -600.0),
        (0.0, -1350.0),
        (1200.0, -600.0),
    ],
}
_CERTIFIED = set()


def certified_layout(name):
    if name not in _CERTIFIED:
        assert coverage_certificate([(0.0, 0.0)] + GRID_LAYOUTS[name])[0], name
        _CERTIFIED.add(name)
    return GRID_LAYOUTS[name]


def route_length(points):
    order = open_route(points)
    route = [points[0]] + [points[i] for i in order]
    return sum(math.dist(a, b) for a, b in zip(route, route[1:]))


def ray_intervals(poly, q, u, failed, clear_radius=20.0):
    """Parameter intervals s>=0 where q+s*u lies in the convex polygon and outside
    every failed clear disk. Returns a sorted list of (s0,s1)."""
    lo, hi = 0.0, math.inf
    for a, b in zip(poly, poly[1:] + poly[:1]):
        ex, ey = b[0] - a[0], b[1] - a[1]
        c0 = ex * (q[1] - a[1]) - ey * (q[0] - a[0])
        c1 = ex * u[1] - ey * u[0]
        if abs(c1) < 1e-12:
            if c0 < -1e-9:
                return []
            continue
        s = -c0 / c1
        if c1 > 0:
            lo = max(lo, s)
        else:
            hi = min(hi, s)
    if hi <= lo:
        return []
    intervals = [(lo, hi)]
    for f in failed:
        dx, dy = q[0] - f[0], q[1] - f[1]
        bq = 2 * (dx * u[0] + dy * u[1])
        cq = dx * dx + dy * dy - clear_radius**2
        disc = bq * bq - 4 * cq
        if disc <= 0:
            continue
        r0, r1 = (-bq - math.sqrt(disc)) / 2, (-bq + math.sqrt(disc)) / 2
        nxt = []
        for a0, a1 in intervals:
            if r1 <= a0 or r0 >= a1:
                nxt.append((a0, a1))
                continue
            if a0 < r0:
                nxt.append((a0, r0))
            if r1 < a1:
                nxt.append((r1, a1))
        intervals = nxt
    return [(a, b) for a, b in intervals if b - a > 1e-6]


class LeanGradientRobot(AllVertexGradientRobot):
    def __init__(
        self,
        *args,
        shared_limit=0,
        layout_choice=True,
        layouts=("rot0", "rot30", "rot15", "rot45"),
        probe_mode="dominant",
        clear_shift=0.0,
        march=True,
        march_steps=4,
        march_span=38.0,
        lateral_gate=100.0,
        **kwargs
    ):
        kwargs.setdefault("gradient_step", 6.0)
        kwargs.setdefault("gradient_axes", "dominant")
        kwargs.setdefault("one_sided", True)
        kwargs.setdefault("grid_search", True)
        super().__init__(*args, shared_limit=shared_limit, **kwargs)
        self.layout_choice = layout_choice
        self.layouts = tuple(layouts)
        self.probe_mode = probe_mode
        self.clear_shift = clear_shift
        self.layout_selected = "rot0"
        self.gradient_ranges = {}
        self.march = march
        self.march_steps = march_steps
        self.march_span = march_span
        self.lateral_gate = lateral_gate
        self.march_clears = 0
        self.march_successes = 0
        for name in self.layouts:
            certified_layout(name)
        self.pending_stations = list(certified_layout("rot0"))

    # ------------------------------------------------------------------ scan
    def scan(self, q, all_known=False, force_discovery=True):
        initial = not self.discovery_stations and math.hypot(*q) < 1e-9
        before = len(self.gradient_diagnostics)
        if self.probe_mode == "perp":
            result = self.scan_perp(
                q, all_known=all_known, force_discovery=force_discovery
            )
        else:
            result = super().scan(
                q, all_known=all_known, force_discovery=force_discovery
            )
        for entry in self.gradient_diagnostics[before:]:
            self.gradient_ranges[entry["channel"]] = math.dist(
                entry["station"], entry["raw_estimate"]
            )
        if initial and self.layout_choice:
            self.choose_layout()
        return result

    def scan_perp(self, q, all_known=False, force_discovery=True):
        """v24 probe logic with one probe point per channel, perpendicular to its bearing."""
        prior = {ch for ch, b in self.beliefs.items() if b.known}
        result = SelectiveScanRobot.scan(
            self, q, all_known=all_known, force_discovery=force_discovery
        )
        if any(abs(v / 150 - round(v / 150)) > 1e-9 for v in q):
            return result
        channels = [
            ch
            for ch, b in self.beliefs.items()
            if ch not in prior and b.known and not b.cleared and b.circle()[1] > 100
        ]
        if not channels:
            return result
        h = self.gradient_step
        bearings = {
            ch: next(
                beta
                for p, beta in reversed(self.beliefs[ch].bearings)
                if math.dist(p, q) < 1e-7
            )
            for ch in channels
        }
        # Visit the probe points in angular order around q to keep movement short.
        current = self.position
        start = (
            math.atan2(current[1] - q[1], current[0] - q[0])
            if math.dist(current, q) > 1e-9
            else 0.0
        )
        order = sorted(
            channels,
            key=lambda ch: (bearings[ch] + math.pi / 2 - start) % (2 * math.pi),
        )
        if self.channel in order:
            order.remove(self.channel)
            order.insert(0, self.channel)
        for ch in order:
            b = self.beliefs[ch]
            if b.cleared:
                continue
            beta = bearings[ch]
            p = (q[0] - h * math.sin(beta), q[1] + h * math.cos(beta))
            r = self.measure(p, ch)
            if r is None:
                r = next(
                    e["response"]
                    for e in reversed(self.log)
                    if e["path"] == "/measure"
                    and e["request"]["channel"] == ch
                    and (e["request"]["position"]["x"], e["request"]["position"]["y"])
                    == p
                )
            if r["measure_result"] != "direction" or b.cleared:
                continue
            change = math.radians((r["svd_deg"] - math.degrees(beta) + 180) % 360 - 180)
            # d(theta)/ds along the left perpendicular equals -1/r.
            if change >= -1e-12:
                continue
            distance = -h / change
            raw = (q[0] + distance * math.cos(beta), q[1] + distance * math.sin(beta))
            if math.hypot(*raw) > 1800:
                continue
            target = project_polygon(raw, b.poly)
            self.gradient_targets[ch] = target
            self.gradient_diagnostics.append(
                dict(
                    channel=ch,
                    station=q,
                    raw_estimate=raw,
                    projected_estimate=target,
                    gradient=[change / h],
                )
            )
        return result

    # ---------------------------------------------------------------- layout
    def choose_layout(self):
        known = [
            self.source_target(ch)
            for ch, b in self.beliefs.items()
            if b.known and not b.cleared
        ]
        best = None
        for name in self.layouts:
            length = route_length([self.position] + known + certified_layout(name))
            if best is None or length < best[0] - 1e-6:
                best = (length, name)
        self.layout_selected = best[1]
        self.pending_stations = list(certified_layout(best[1]))

    # ------------------------------------------------------------- clearing
    def finish_source(self, ch):
        b = self.beliefs[ch]
        if ch in self.gradient_targets and not b.cleared:
            target = self.source_target(ch)
            if self.clear_shift > 0:
                r_est = self.gradient_ranges.get(ch, 1e9)
                shift = min(self.clear_shift, 0.5 * (20 - 0.05 * r_est))
                d = math.dist(self.position, target)
                if shift > 0 and d > shift:
                    target = project_polygon(
                        (
                            target[0] + (self.position[0] - target[0]) * shift / d,
                            target[1] + (self.position[1] - target[1]) * shift / d,
                        ),
                        b.poly,
                    )
            del self.gradient_targets[ch]
            self.gradient_guesses += 1
            if self.clear(target, ch, safe=False):
                self.gradient_first_clear_successes += 1
                return
            self.measure(self.position, ch)
            if b.cleared:
                return
            if self.march:
                self.march_source(ch)
                if b.cleared:
                    return
        return self._conservative_finish(ch)

    def _conservative_finish(self, ch):
        # Skip VertexGradientRobot.finish_source (its guess is consumed) and run the
        # original conservative chain: Selective -> Combined -> Event -> Approach -> Routing -> Robot.
        return SelectiveScanRobot.finish_source(self, ch)

    def march_source(self, ch):
        """After a failed speculative clear: march along the fresh near bearing.

        The polygon is still the only certificate. Each step clears at the first
        feasible ray interval; long remaining intervals trigger one lateral bearing
        so the two near wedges cross and bound the range.
        """
        b = self.beliefs[ch]
        for _ in range(self.march_steps):
            if b.cleared or b.circle()[1] < 19.95:
                return
            here = self.position
            recent = [(p, beta) for p, beta in b.bearings if math.dist(p, here) < 1e-7]
            if not recent:
                return
            _, beta = recent[-1]
            u = (math.cos(beta), math.sin(beta))
            intervals = ray_intervals(b.poly, here, u, b.failed)
            if not intervals:
                return
            a0, a1 = intervals[0]
            if a1 - a0 <= self.march_span:
                s = (a0 + a1) / 2
            else:
                s = a0 + self.march_span / 2
            point = (here[0] + s * u[0], here[1] + s * u[1])
            self.march_clears += 1
            if self.clear(point, ch, safe=False):
                self.march_successes += 1
                return
            remaining = sum(max(0.0, hi - max(lo, s + 20)) for lo, hi in intervals)
            if remaining > self.lateral_gate:
                # One transverse bearing: crossing wedges bound the range.
                offset = 35.0
                side = (point[0] - offset * u[1], point[1] + offset * u[0])
                self.measure(side, ch)
            else:
                self.measure(point, ch)
            if b.cleared:
                return

    def run(self, resume=False):
        result = super().run(resume=resume)
        result.update(
            layout_selected=self.layout_selected,
            march_clears=self.march_clears,
            march_successes=self.march_successes,
        )
        return result
