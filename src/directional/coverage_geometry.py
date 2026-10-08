"""Continuous directional coverage proof, purely geometric, no simulation.

Each accepted square lies inside the convex hull of stations whose entire
square-distance bound is below 1000 m. Hence every possible closed half-plane
through every point in the square contains an in-range station.
"""

import math


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return []
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


def certify(stations, min_cell=0.25):
    stack = [(0.0, 0.0, 1800.0)]
    checked = 0
    leaves = []
    while stack:
        x, y, h = stack.pop()
        checked += 1
        if math.hypot(max(abs(x) - h, 0), max(abs(y) - h, 0)) > 1800 + 1e-7:
            continue
        near = [
            q
            for q in stations
            if math.hypot(abs(q[0] - x) + h, abs(q[1] - y) + h) < 1000 - 1e-6
        ]
        poly = hull(near)
        corners = [(x + dx * h, y + dy * h) for dx in (-1, 1) for dy in (-1, 1)]
        if len(poly) >= 3 and all(
            cross(a, b, p) >= 1e-6
            for a, b in zip(poly, poly[1:] + poly[:1])
            for p in corners
        ):
            leaves.append((x, y, h))
            continue
        if 2 * h <= min_cell:
            return dict(certified=False, checked=checked, unresolved=[x, y, h])
        half = h / 2
        stack.extend(
            (x + dx * half, y + dy * half, half) for dx in (-1, 1) for dy in (-1, 1)
        )
    return dict(certified=True, checked=checked, leaf_count=len(leaves))


def rings(
    inner_count=6, outer_count=12, inner_radius=950.0, outer_radius=1865.0, offset=0.0
):
    return (
        [(0.0, 0.0)]
        + [
            (
                inner_radius * math.cos(2 * math.pi * i / inner_count + offset),
                inner_radius * math.sin(2 * math.pi * i / inner_count + offset),
            )
            for i in range(inner_count)
        ]
        + [
            (
                outer_radius * math.cos(2 * math.pi * i / outer_count),
                outer_radius * math.sin(2 * math.pi * i / outer_count),
            )
            for i in range(outer_count)
        ]
    )


if __name__ == "__main__":
    import json

    for count in (6, 8, 10, 12):
        q = rings(inner_count=count)
        print(
            json.dumps(
                dict(stations=len(q), inner=count, proof=certify(q)), ensure_ascii=False
            ),
            flush=True,
        )
