"""Speculative clearance positions account for both incoming and outgoing legs.

The soft disk is a route heuristic only, not a confidence or clearance proof.
Every attempt is marked unsafe and relies on the real clear response. The
original conservative envelope and failed-attempt recovery remain untouched.
"""

import math
from policy import OfficialQ4Robot
from gradient import project_polygon


def best_clear_point(center, radius, start, end, poly):
    if radius <= 0:
        return center

    def value(p):
        return math.dist(start, p) + (math.dist(p, end) if end is not None else 0.0)

    candidates = [center]
    if end is None:
        d = math.dist(start, center)
        if d > 1e-9:
            candidates.append(
                tuple(
                    center[a] + (start[a] - center[a]) * min(radius, d) / d
                    for a in (0, 1)
                )
            )
    else:
        # If the direct travel chord intersects the soft disk, use a point on
        # the chord; polygon projection preserves only hard feasible geometry.
        v = (end[0] - start[0], end[1] - start[1])
        den = v[0] * v[0] + v[1] * v[1]
        t = (
            max(
                0.0,
                min(
                    1.0,
                    ((center[0] - start[0]) * v[0] + (center[1] - start[1]) * v[1])
                    / den,
                ),
            )
            if den
            else 0.0
        )
        point = (start[0] + t * v[0], start[1] + t * v[1])
        if math.dist(point, center) <= radius:
            candidates.append(point)

    def point(angle):
        return (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        )

    angles = [i * math.tau / 64 for i in range(64)]
    theta = min(angles, key=lambda t: value(point(t)))
    low, high = theta - math.tau / 64, theta + math.tau / 64
    for _ in range(24):
        a = (2 * low + high) / 3
        b = (low + 2 * high) / 3
        if value(point(a)) <= value(point(b)):
            high = b
        else:
            low = a
    candidates.append(point((low + high) / 2))
    candidates = [project_polygon(p, poly) for p in candidates]
    winner = min(candidates, key=value)
    assert math.dist(winner, center) <= radius + 1e-6
    assert value(winner) <= value(center) + 1e-6
    return winner


class ClearNeighborhood(OfficialQ4Robot):
    constant_radius = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_neighborhood_diagnostics = []

    def finish_source(self, ch):
        b = self.beliefs[ch]
        if ch in self.gradient_targets and not b.cleared:
            original = self.source_target(ch)
            r = self.gradient_ranges.get(ch, 1e9)
            radius = (
                max(0.0, min(15.0, 20.0 - 0.025 * r))
                if self.constant_radius is None
                else self.constant_radius
            )
            end = None
            for kind, key in self.route_keys:
                if kind == "source":
                    if key != ch and not self.beliefs[key].cleared:
                        end = self.source_target(key)
                        break
                elif key in self.pending_stations:
                    end = key
                    break
            target = best_clear_point(original, radius, self.position, end, b.poly)
            self.gradient_targets[ch] = target
            self.clear_neighborhood_diagnostics.append(
                dict(
                    channel=ch,
                    range_estimate_m=r,
                    soft_radius_m=radius,
                    start=self.position,
                    outgoing=end,
                    original=original,
                    chosen=target,
                    shift_m=math.dist(target, original),
                    assumption_only=True,
                )
            )
        return super().finish_source(ch)

    def run(self, resume=False):
        result = super().run(resume)
        result["clear_neighborhood_diagnostics"] = self.clear_neighborhood_diagnostics
        return result


class ClearNeighborhood10(ClearNeighborhood):
    constant_radius = 10.0
