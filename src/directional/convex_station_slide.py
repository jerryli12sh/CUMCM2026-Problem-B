"""Optimize a single scan site's convex replacement region, then certify it.

Default mode keeps complete lattice vertices fixed. Mode 'all' also permits
moving their discovery scan, while keeping the original lattice probe anchor.
Geometry proposes points; the full regional cost, including discovery
and anchor fees, decides adoption. All changes require a fresh integer proof.
"""

import math, time
from fractions import Fraction as F
import numpy as np
from scipy.optimize import minimize_scalar
from station_mobility_bound import region, clip, round_inside, sees, RADIUS
from coverage_adversary import adversary
from fast_coverage_reject import certify, verdict
from certify_layout_integer import certify as final_python_certify
from fixed_order_cost import Cost


def optimum(poly, start, anchor, end, probability):
    vertices = [tuple(float(v) / 1000 for v in p) for p in poly]

    def value(q):
        return (
            math.dist(start, q)
            + probability * math.dist(anchor, q)
            + (0.0 if end is None else (1 - probability) * math.dist(end, q))
        )

    inside = all(
        (b[0] - a[0]) * (start[1] - a[1]) - (b[1] - a[1]) * (start[0] - a[0]) >= -1e-10
        for a, b in zip(vertices, vertices[1:] + vertices[:1])
    )
    candidates = [start] if inside else []
    for a, b in zip(vertices, vertices[1:] + vertices[:1]):

        def point(t):
            return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

        result = minimize_scalar(
            lambda t: value(point(t)),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-7},
        )
        candidates.extend((a, b, point(float(result.x))))
    return min(candidates, key=value)


def separate(points, i, candidate, poly):
    test = points.copy()
    test[i] = list(candidate)
    others = points[:i] + points[i + 1 :]
    g, u, _ = adversary(np.asarray(test) / 1000000, domain=1.77, refine=True)
    added = 0
    for gg, uu in zip(g, u):
        gg = round_inside(gg * 1000000)
        uu = tuple(map(int, np.rint(uu * 1000000)))
        if uu == (0, 0) or any(sees(p, gg, uu) for p in test):
            continue
        assert not any(sees(p, gg, uu) for p in others) and sees(points[i], gg, uu)
        poly = clip(poly, (-uu[0], -uu[1], -gg[0] * uu[0] - gg[1] * uu[1]))
        added += 1
        v = (candidate[0] - gg[0], candidate[1] - gg[1])
        norm2 = RADIUS * RADIUS * (v[0] * v[0] + v[1] * v[1])
        support = math.isqrt(norm2)
        if support * support < norm2:
            support += 1
        if any(v):
            poly = clip(poly, (v[0], v[1], v[0] * gg[0] + v[1] * gg[1] + support))
    return poly, added


def select(bot, task):
    if task[0] != "scan":
        return task
    old = tuple(task[2])
    node = tuple(float(150 * round(v / 150)) for v in old)
    was_vertex = math.dist(old, node) < 1e-7
    if was_vertex and bot.convex_slide != "all":
        return task
    if not bot.route_keys or bot.route_keys[0] != ("scan", old):
        return task
    key = (tuple(bot.required_stations), old, bot.position, tuple(bot.route_keys))
    if key == getattr(bot, "_convex_slide_key", None):
        return task
    bot._convex_slide_key = key
    start = time.monotonic()
    model = Cost(bot)
    base = model.first(old)
    if not math.isfinite(base):
        return task
    i = bot.required_stations.index(old)
    points = [[int(round(v * 1000)) for v in p] for p in bot.required_stations]
    data = region(points, i)
    poly = [tuple(map(F, p)) for p in data["outer_polygon_mm"]]
    for axis in (0, 1):
        v = [0, 0]
        v[axis] = 1
        poly = clip(poly, (*v, int(round(node[axis] * 1000)) + 75000))
        v[axis] = -1
        poly = clip(poly, (*v, -int(round(node[axis] * 1000)) + 75000))
    assert len(poly) >= 3
    end = tuple(model.points[1]) if len(model.points) > 1 else None
    probability = model.first_occupancy()
    best = base
    chosen = None
    proof = None
    attempts = 0
    cuts = 0
    seen = set()
    records = []

    def check(q):
        nonlocal attempts, best, chosen, proof
        candidate = points.copy()
        candidate[i] = list(q)
        attempts += 1
        provisional = verdict(candidate, domain_mm=1770000, max_depth=22)
        certificate = (
            certify(candidate, domain_mm=1770000, max_depth=22)
            if provisional is None
            else dict(certified=provisional)
        )
        if certificate["certified"]:
            p = tuple(v / 1000 for v in q)
            score = model.first(p)
            if score < best - 1e-9:
                best = score
                chosen = p
                proof = certificate
        return certificate["certified"]

    for iteration in range(6):
        q = optimum(poly, bot.position, node, end, probability)
        q = tuple(int(round(v * 1000)) for v in q)
        # Preserve the old nearest anchor, including Python's boundary tie rule.
        q = tuple(
            min(int(node[a] * 1000) + 74999, max(int(node[a] * 1000) - 74999, q[a]))
            for a in (0, 1)
        )
        if q in seen or math.dist(q, points[i]) < 100:
            break
        seen.add(q)
        accepted = check(q)
        records.append(dict(iteration=iteration, candidate_mm=q, certified=accepted))
        if accepted:
            break
        lo = 0.0
        hi = 1.0
        for _ in range(8):
            t = (lo + hi) / 2
            mid = tuple(round(points[i][a] + t * (q[a] - points[i][a])) for a in (0, 1))
            if check(mid):
                lo = t
            else:
                hi = t
        records[-1]["certified_line_fraction"] = lo
        if lo > 0.98:
            break
        poly, added = separate(points, i, q, poly)
        cuts += added
        if not added or len(poly) < 3:
            break
    taken = chosen is not None and best < base - 1.0
    diagnostic = dict(
        action_count=len(bot.log),
        old=old,
        chosen=chosen if taken else old,
        nearest_anchor=node,
        moved_original_vertex=was_vertex and taken,
        expected_objective_before=base,
        expected_objective_after=best if taken else base,
        predicted_saving_s=base - best,
        adopted=taken,
        certificate_attempts=attempts,
        additional_witnesses=cuts,
        records=records,
        elapsed_s=time.monotonic() - start,
    )
    bot.convex_slide_diagnostics.append(diagnostic)
    if not taken:
        return task
    final_points = points.copy()
    final_points[i] = [round(v * 1000) for v in chosen]
    proof = final_python_certify(final_points, domain_mm=1770000, max_depth=22)
    if not proof["certified"]:
        diagnostic.update(adopted=False, chosen=old, final_proof_failed=True)
        return task
    assert proof["certified"] and tuple(150 * round(v / 150) for v in chosen) == node
    bot.required_stations = [chosen if p == old else p for p in bot.required_stations]
    bot.pending_stations = [chosen if p == old else p for p in bot.pending_stations]
    bot.route_keys = [
        ("scan", chosen) if k == ("scan", old) else k for k in bot.route_keys
    ]
    if bot.next_scan == old:
        bot.next_scan = chosen
    bot.geometry_certificate = proof
    return ("scan", task[1], chosen)
