"""Exact integer-arithmetic coverage certificate for millimetre station points.

All comparisons are integer squared distances and oriented areas. The proof
uses a dyadic quadtree, without floating point hulls or sampling acceptance.
"""

import argparse, json, math, time
from pathlib import Path


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return []

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    low = []
    high = []
    for p in points:
        while len(low) > 1 and cross(low[-2], low[-1], p) <= 0:
            low.pop()
        low.append(p)
    for p in reversed(points):
        while len(high) > 1 and cross(high[-2], high[-1], p) <= 0:
            high.pop()
        high.append(p)
    return low[:-1] + high[:-1]


def certify(points_mm, domain_mm=1800000, range_mm=1000000, max_depth=23, record=False):
    P = [tuple(map(int, p)) for p in points_mm]
    h = domain_mm
    stack = [(0, 0, 0)]
    count = 0
    leaves = []
    while stack:
        x, y, level = stack.pop()
        count += 1
        scale = 1 << level
        nx = max(abs(x) - h, 0)
        ny = max(abs(y) - h, 0)
        if nx * nx + ny * ny > (domain_mm * scale) ** 2:
            continue
        near = []
        for j, (a, b) in enumerate(P):
            dx = abs(a * scale - x) + h
            dy = abs(b * scale - y) + h
            if dx * dx + dy * dy <= (range_mm * scale) ** 2:
                near.append((a, b))
        poly = hull(near)
        good = len(poly) >= 3
        if good:
            for a, b in zip(poly, poly[1:] + poly[:1]):
                for dx, dy in ((-h, -h), (-h, h), (h, -h), (h, h)):
                    if (b[0] - a[0]) * (y + dy - a[1] * scale) - (b[1] - a[1]) * (
                        x + dx - a[0] * scale
                    ) < 0:
                        good = False
                        break
                if not good:
                    break
        if good:
            entry = [x, y, level]
            leaves.append(entry if record else None)
            continue
        if level >= max_depth:
            return dict(
                certified=False,
                arithmetic="exact integers",
                checked=count,
                domain_mm=domain_mm,
                range_mm=range_mm,
                unresolved_center_m=[x / scale / 1000, y / scale / 1000],
                unresolved_halfwidth_m=h / scale / 1000,
                max_depth=max_depth,
            )
        for dx, dy in ((-h, -h), (-h, h), (h, -h), (h, h)):
            stack.append((2 * x + dx, 2 * y + dy, level + 1))
    result = dict(
        certified=True,
        arithmetic="exact integers",
        checked=count,
        leaf_count=len(leaves),
        domain_mm=domain_mm,
        range_mm=range_mm,
        points_mm=P,
        max_depth=max_depth,
    )
    if record:
        result["leaves"] = leaves
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output")
    ap.add_argument("--domain", type=float, default=1800)
    args = ap.parse_args()
    d = json.loads(Path(args.input).read_text())
    p = [[round(a * 1000), round(b * 1000)] for a, b in d["points"]]
    began = time.monotonic()
    proof = certify(p, domain_mm=round(args.domain * 1000), record=True)
    proof["elapsed_s"] = time.monotonic() - began
    if args.output:
        Path(args.output).write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in proof.items() if k not in ("leaves", "points_mm")})
    )


if __name__ == "__main__":
    main()
