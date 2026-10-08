"""Conservative cell exclusion for the union of omni and directional types.

No probabilistic pruning. A cell is discarded only when every location in it
has an in-range negative (ruling out omni) and no directional orientation can
satisfy necessary expanded angular constraints. Unresolved cells are retained.
"""

import math
from geometry import clip, hull

TAU = 2 * math.pi
ANGULAR_EPS = 1e-9
DISTANCE_EPS = 1e-6


def area(poly):
    return (
        abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(poly, poly[1:] + poly[:1])))
        / 2
    )


def arc(center, halfwidth):
    if halfwidth >= math.pi - ANGULAR_EPS:
        return [(0.0, TAU)]
    lo = (center - halfwidth) % TAU
    hi = (center + halfwidth) % TAU
    return [(lo, hi)] if lo <= hi else [(0.0, hi), (lo, TAU)]


def intersect_intervals(a, b):
    out = []
    for lo, hi in a:
        for start, end in b:
            x, y = max(lo, start), min(hi, end)
            if x <= y + ANGULAR_EPS:
                out.append((x - ANGULAR_EPS, y + ANGULAR_EPS))
    return out


def cell_possible(poly, positive, negative):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    center = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    radius = math.hypot(max(xs) - min(xs), max(ys) - min(ys)) / 2 + DISTANCE_EPS
    # For every g in this cell, R >= max(1000, max_p |p-g|).
    lower_radius = max(
        [1000.0] + [max(0.0, math.dist(p, center) - radius) for p in positive]
    )
    unavoidable = [
        p
        for p in negative
        if math.dist(p, center) + radius < lower_radius - DISTANCE_EPS
    ]
    if not unavoidable:
        # Omni might still be feasible; positive directions alone cannot be
        # restricted to a single half-plane for the union of source types.
        return True
    intervals = [(0.0, TAU)]
    for points, opposite in ((positive, False), (unavoidable, True)):
        for p in points:
            dx, dy = p[0] - center[0], p[1] - center[1]
            distance = math.hypot(dx, dy)
            if distance <= radius + DISTANCE_EPS:
                continue
            spread = math.asin(min(1.0, radius / distance))
            angle = math.atan2(dy, dx) + (math.pi if opposite else 0.0)
            halfwidth = math.pi / 2 + spread + ANGULAR_EPS
            intervals = intersect_intervals(intervals, arc(angle, halfwidth))
            if not intervals:
                return False
    return True


def prune_orientation_envelope(poly, positive, negative, max_cells=511, min_width=5.0):
    original = list(poly)
    before = area(original)
    if not positive or not negative or len(poly) < 3 or before < 1e-5:
        return original, dict(
            checked=0,
            rejected_cells=0,
            area_before=before,
            area_after=before,
            applied=False,
            budget_exhausted=False,
        )
    stack = [original]
    kept = []
    checked = 0
    rejected = 0
    while stack and checked < max_cells:
        cell = stack.pop()
        checked += 1
        if not cell_possible(cell, positive, negative):
            rejected += 1
            continue
        xs = [p[0] for p in cell]
        ys = [p[1] for p in cell]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if max(width, height) <= min_width:
            kept.extend(cell)
            continue
        if width >= height:
            mid = (min(xs) + max(xs)) / 2
            children = (clip(cell, 1.0, 0.0, mid), clip(cell, -1.0, 0.0, -mid))
        else:
            mid = (min(ys) + max(ys)) / 2
            children = (clip(cell, 0.0, 1.0, mid), clip(cell, 0.0, -1.0, -mid))
        for child in children:
            if child:
                stack.append(child)
    exhausted = bool(stack)
    for cell in stack:
        kept.extend(cell)
    candidate = hull(kept)
    after = area(candidate) if len(candidate) >= 3 else before
    applied = bool(len(candidate) >= 3 and after < before - 1e-5)
    # Empty/degenerate outcomes are not adopted. This defensive fallback keeps
    # the prior conservative envelope rather than claiming source absence.
    if not applied:
        candidate = original
        after = before
    return candidate, dict(
        checked=checked,
        rejected_cells=rejected,
        area_before=before,
        area_after=after,
        applied=applied,
        budget_exhausted=exhausted,
    )
