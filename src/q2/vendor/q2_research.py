"""Q2 follow-up: adaptive bearing enclosure, D/MEC radius, threshold maps.
Run with workspace .venv/bin/python. Original scripts are imported, never edited.
Floating point set-inclusion bounds; no outward-rounded machine certification.
"""

from pathlib import Path
import math, json, heapq, itertools, time, argparse
import numpy as np
from scipy.spatial import ConvexHull
import q2_all_outcomes as old

ROOT = Path(__file__).resolve().parent
A = old.ALPHA


def configure(alpha_deg=1, first_station=(0.0, 0.0), bearing_deg=0.0):
    global A
    A = math.radians(alpha_deg)
    old.ALPHA = A
    old.base.ALPHA = A
    old.OUTER, old.INNER = old.base.priors(96)
    old.OUTER = old.clip(old.OUTER, -1, 0, -5 * math.cos(A))
    # Transform target-disk center into the local (bearing, perpendicular) frame.
    theta = math.radians(bearing_deg)
    sx, sy = first_station
    center = (
        -sx * math.cos(theta) - sy * math.sin(theta),
        sx * math.sin(theta) - sy * math.cos(theta),
    )
    if first_station != (0.0, 0.0):
        old.OUTER = old.cap_disk(old.OUTER, center, 1800, True, sides=2048)
        old.INNER = old.cap_disk(old.INNER, center, 1800, False, sides=2048)


configure()


def mec(points):
    """Incremental smallest enclosing circle; hull first reduces redundant points."""
    if not points:
        return (0.0, (0.0, 0.0))
    p = list(dict.fromkeys(points))
    if len(p) > 8:
        try:
            p = [p[i] for i in ConvexHull(p).vertices]
        except Exception:
            pass
    # Fixed permutation only for runtime, not a statistical approximation.
    np.random.default_rng(42).shuffle(p)
    c = p[0]
    r = 0.0
    for i, a in enumerate(p):
        if math.dist(a, c) <= r + 1e-9:
            continue
        c = a
        r = 0.0
        for j, b in enumerate(p[:i]):
            if math.dist(b, c) <= r + 1e-9:
                continue
            c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            r = math.dist(a, b) / 2
            for d in p[:j]:
                if math.dist(d, c) <= r + 1e-9:
                    continue
                bx, by = b[0] - a[0], b[1] - a[1]
                dx, dy = d[0] - a[0], d[1] - a[1]
                det = 2 * (bx * dy - by * dx)
                if abs(det) < 1e-12:
                    u, v = max(
                        itertools.combinations([a, b, d], 2),
                        key=lambda uv: math.dist(*uv),
                    )
                    c = ((u[0] + v[0]) / 2, (u[1] + v[1]) / 2)
                    r = math.dist(u, v) / 2
                else:
                    bn = bx * bx + by * by
                    dn = dx * dx + dy * dy
                    c = (
                        a[0] + (bn * dy - dn * by) / det,
                        a[1] + (bx * dn - dx * bn) / det,
                    )
                    r = math.dist(a, c)
    return r, c


def metrics(p, radius=True):
    d = old.diameter(p)
    return np.array([d, mec(p)[0] if radius else d / 2])


def certificate(q, tol=0.1, cutoff=math.inf, radius=True, max_nodes=6000):
    q = tuple(q)
    if math.hypot(*q) < 1e-9:
        l = metrics(old.INNER, radius)
        u = metrics(old.OUTER, radius)
        return dict(q=q, lower=l.tolist(), upper=u.tolist(), nodes=0, converged=True)
    c = sum(v * v for v in q) / 2
    vo = old.outside_vertices(old.clip(old.OUTER, *q, c + 1e-7), q, 1000 - 1e-7)
    vi = old.outside_vertices(old.clip(old.INNER, *q, c - 1e-7), q, 1000 + 1e-7)
    sl, su = metrics(vi, radius), metrics(vo, radius)
    lower = sl.copy()
    fixed = np.maximum(su, [10, 5])
    if lower[0] > cutoff:
        return dict(
            q=q, lower=lower.tolist(), upper=[None, None], nodes=0, converged=False
        )
    po = old.cap_disk(old.OUTER, q, 1500, True, sides=512)
    pi = old.cap_disk(old.INNER, q, 1500, False, sides=512)
    start, width = old.angle_range(po, q)
    if width == 0:
        return dict(
            q=q, lower=lower.tolist(), upper=fixed.tolist(), nodes=0, converged=True
        )
    heap = []
    counter = itertools.count()
    nodes = 0
    worst_beta = 0

    def add(lo, hi):
        nonlocal lower, nodes, worst_beta
        b = (lo + hi) / 2
        pu = old.outside_vertices(
            old.posterior(po, q, b, A + (hi - lo) / 2), q, 5 - 1e-7
        )
        pl = old.outside_vertices(old.posterior(pi, q, b, A), q, 5 + 1e-7)
        u = metrics(pu, radius)
        l = metrics(pl, radius)
        if l[0] > lower[0]:
            worst_beta = b
        lower = np.maximum(lower, l)
        nodes += 1
        heapq.heappush(heap, (-max(u[0], 2 * u[1]), next(counter), lo, hi, u))

    for i in range(16):
        add(start + width * i / 16, start + width * (i + 1) / 16)
    while heap and nodes < max_nodes:
        upper = np.maximum(fixed, np.max([x[4] for x in heap], axis=0))
        if np.max((upper - lower) * [1, 2]) <= tol or lower[0] > cutoff:
            break
        # Refine a node that can still determine either objective.
        candidates = [
            (max(x[4][0] - lower[0], 2 * (x[4][1] - lower[1])), i)
            for i, x in enumerate(heap)
        ]
        _, i = max(candidates)
        entry = heap[i]
        heap[i] = heap[-1]
        heap.pop()
        heapq.heapify(heap)
        _, _, lo, hi, _ = entry
        mid = (lo + hi) / 2
        add(lo, mid)
        add(mid, hi)
    upper = np.maximum(fixed, np.max([x[4] for x in heap], axis=0))
    return dict(
        q=q,
        lower=lower.tolist(),
        upper=upper.tolist(),
        silent_lower=sl.tolist(),
        silent_upper=su.tolist(),
        nodes=nodes,
        converged=bool(np.max((upper - lower) * [1, 2]) <= tol),
        worst_sampled_bearing_deg=math.degrees(worst_beta) % 360,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "grid"], default="check")
    args = ap.parse_args()
    if args.mode == "check":
        result = [
            certificate(q, 0.02)
            for q in [(961, 512), (950, 500), (834, 552), (750, 600)]
        ]
        (ROOT / "research_certificates.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    else:
        # Full informative bounding box in upper half-plane. Symmetry gives lower half.
        axes = [np.arange(-1500, 3000.1, 20), np.arange(0, 1540.1, 20)]
        rows = []
        t = time.time()
        for x in axes[0]:
            for y in axes[1]:
                c = certificate((float(x), float(y)), 0.25, 160, radius=False)
                rows.append([float(x), float(y), c["lower"][0], c["upper"][0]])
            if int(x) % 200 == 0:
                print("grid", x, len(rows), round(time.time() - t, 1), flush=True)
        (ROOT / "research_grid_coarse.json").write_text(
            json.dumps(
                dict(
                    step=20,
                    scope="full bounding box; b>=0; D only; null upper means excluded by lower >160",
                    rows=rows,
                ),
                separators=(",", ":"),
            )
        )
        rows = []
        for x in np.arange(650, 1100.1, 5):
            for y in np.arange(250, 850.1, 5):
                c = certificate((float(x), float(y)), 0.15, 130, radius=False)
                rows.append([float(x), float(y), c["lower"][0], c["upper"][0]])
            if int(x) % 50 == 0:
                print("fine", x, len(rows), round(time.time() - t, 1), flush=True)
        (ROOT / "research_grid_fine.json").write_text(
            json.dumps(
                dict(
                    step=5,
                    scope="local window; D only; null upper means excluded by lower >130",
                    rows=rows,
                ),
                separators=(",", ":"),
            )
        )


if __name__ == "__main__":
    main()
