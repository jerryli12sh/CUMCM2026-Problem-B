"""Necessary one-station replacement regions with exact rational witnesses.

Remove station i. Any integer source/heading pair unseen by every other site
must be seen by its replacement: a1000m disk and a receiving half-plane.
Intersect these necessary constraints using outer disk support half-planes.
All visibility predicates and polygon clipping use integer/rational arithmetic.
The region contains every valid SINGLE-site replacement, with other sites
fixed. It is not a bound for simultaneous layout changes or the entire mission.
"""

import argparse, json, math, shutil, time
from fractions import Fraction as F
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from coverage_adversary import adversary

HERE = Path(__file__).resolve().parent
DOMAIN = 1770000
RADIUS = 1000000


def round_inside(p):
    x, y = map(lambda z: int(round(z)), p)
    while x * x + y * y > DOMAIN * DOMAIN:
        if abs(x) >= abs(y):
            x -= 1 if x > 0 else -1
        else:
            y -= 1 if y > 0 else -1
    return x, y


def sees(p, g, u):
    x = p[0] - g[0]
    y = p[1] - g[1]
    return x * x + y * y <= RADIUS * RADIUS and x * u[0] + y * u[1] >= 0


def clip(poly, plane):
    if not poly:
        return []
    A, B, C = map(int, plane)
    out = []
    for p, q in zip(poly, poly[1:] + poly[:1]):
        a = A * p[0] + B * p[1] - C
        b = A * q[0] + B * q[1] - C
        if a <= 0:
            out.append(p)
        if (a < 0 and b > 0) or (a > 0 and b < 0):
            t = a / (a - b)
            out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
    result = []
    for p in out:
        if not result or p != result[-1]:
            result.append(p)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def region(points, i):
    others = points[:i] + points[i + 1 :]
    g, u, diag = adversary(
        np.asarray(others) / 1000000, domain=1.77, dense=True, refine=True
    )
    witnesses = []
    for gg, uu in zip(g, u):
        gg = round_inside(gg * 1000000)
        uu = tuple(map(int, np.rint(uu * 1000000)))
        if uu == (0, 0) or any(sees(p, gg, uu) for p in others):
            continue
        assert sees(points[i], gg, uu)
        witnesses.append((gg, uu))
    witnesses = list(dict.fromkeys(witnesses))
    assert witnesses, "No sole-observer witness found"
    planes = [(-v[0], -v[1], -v[0] * g[0] - v[1] * g[1]) for g, v in witnesses]
    # Every disk support bound is rounded OUTWARD using exact integer sqrt.
    for angle in np.arange(128) * math.tau / 128:
        v = (
            int(round(1000000 * math.cos(angle))),
            int(round(1000000 * math.sin(angle))),
        )
        norm2 = RADIUS * RADIUS * (v[0] * v[0] + v[1] * v[1])
        support = math.isqrt(norm2)
        if support * support < norm2:
            support += 1
        rhs = min(v[0] * g[0] + v[1] * g[1] for g, _ in witnesses) + support
        planes.append((v[0], v[1], rhs))
    planes = list(dict.fromkeys(planes))
    poly = [
        (F(-3000000), F(-3000000)),
        (F(3000000), F(-3000000)),
        (F(3000000), F(3000000)),
        (F(-3000000), F(3000000)),
    ]
    for plane in planes:
        poly = clip(poly, plane)
    assert len(poly) >= 3 and all(
        A * points[i][0] + B * points[i][1] <= C for A, B, C in planes
    )
    largest = max((p[0] - points[i][0]) ** 2 + (p[1] - points[i][1]) ** 2 for p in poly)
    ceiling = (largest.numerator + largest.denominator - 1) // largest.denominator
    upper = math.isqrt(ceiling)
    if upper * upper < ceiling:
        upper += 1
    return dict(
        station_index=i,
        station_mm=points[i],
        witnesses=[dict(source_mm=g, normal=u) for g, u in witnesses],
        outer_halfplanes=planes,
        outer_polygon_mm=[[str(x), str(y)] for x, y in poly],
        single_site_max_displacement_upper_mm=upper,
        polygon_vertices=len(poly),
        witness_count=len(witnesses),
        adversary_diagnostic=diag,
    )


def job(spec):
    name, points, i = spec
    start = time.monotonic()
    value = region(points, i)
    value.update(layout=name, elapsed_s=time.monotonic() - start)
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="single_station_mobility")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    out = HERE / args.name
    out.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__), out / "source_snapshot.py")
    library = json.loads((HERE / "catalog_lattice20.json").read_text())
    layouts = [x for x in library if str(x["degrees"]).endswith("_m0_r0")]
    assert len(layouts) == 6
    jobs = [
        (x["degrees"], [[int(round(v * 1000)) for v in p] for p in x["points"]], i)
        for x in layouts
        for i in range(1, 20)
    ]
    (out / "plan.json").write_text(
        json.dumps(
            dict(
                no_scene_inputs=True,
                layouts=layouts,
                stations_per_layout=19,
                meaning="Only one site moves; all other19 sites fixed. Necessary regions, not sufficient coverage.",
            ),
            indent=2,
        )
    )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(job, jobs):
            rows.append(result)
            (out / f"{result['layout']}_{result['station_index']:02d}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "layout",
                            "station_index",
                            "single_site_max_displacement_upper_mm",
                            "witness_count",
                            "elapsed_s",
                        )
                    }
                ),
                flush=True,
            )
    summary = {
        x["degrees"]: dict(
            max_displacement_upper_m=[
                r["single_site_max_displacement_upper_mm"] / 1000
                for r in rows
                if r["layout"] == x["degrees"]
            ]
        )
        for x in layouts
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
