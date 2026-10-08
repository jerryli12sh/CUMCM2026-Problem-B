"""Place known-source visits according to future regional discovery, softly.

The objective prices actual scan fees and optimistic later source insertions.
Known-source visits can be inserted AFTER the scans that reveal their neighbors.
No posterior mass replaces the existing full-domain completion certificate.
"""

import math
import numpy as np
from route_optimizer import optimize_route
from predictive_route import survival_table
from rollout_type_prior import condition


def select(bot, baseline):
    if len(bot.pending_stations) < 2:
        return baseline
    survival_table(bot, [])
    h = condition(bot)
    if h["expected"] < 0.5:
        return baseline
    n0 = len(bot.particle_cache[0])
    xy, radius, normals = bot.particle_cache
    known = sum(b.known or b.cleared for b in bot.beliefs.values())
    eo = h["expected_omni"]
    ed = h["expected_directional"]
    ids = np.flatnonzero(h["alive"])
    oi = ids[ids < n0]
    di = ids[ids >= n0]
    oi = oi[:: max(1, len(oi) // 256)]
    di = di[:: max(1, len(di) // 256)]
    ids = np.r_[oi, di]
    ghost = xy[ids % n0]
    gr = radius[ids % n0]
    gn = normals[ids % n0]
    isdir = ids >= n0
    weights = np.r_[
        np.full(len(oi), eo / max(1, len(oi))), np.full(len(di), ed / max(1, len(di)))
    ]
    tasks = [
        ("source", c, bot.source_target(c))
        for c, b in bot.beliefs.items()
        if b.known and not b.cleared
    ]
    source_count = len(tasks)
    tasks += [("scan", i, p) for i, p in enumerate(bot.pending_stations)]
    keys = [("source", c) if k == "source" else ("scan", p) for k, c, p in tasks]
    points = np.array([p for _, _, p in tasks])
    full = np.vstack((bot.position, points))
    n = len(points)
    dd = np.linalg.norm(full[:, None, :] - full[None, :, :], axis=2) / 5
    dg = np.linalg.norm(points[:, None, :] - ghost[None, :, :], axis=2) / 5
    delta = points[:, None, :] - ghost[None, :, :]
    vis = (np.sum(delta * delta, axis=2) <= gr**2) & (
        ~isdir | (np.sum(delta * gn, axis=2) >= 0)
    )
    vis[:source_count] = False
    blank = 20 - known - h["expected"]
    cache = {}
    nodes = np.round(points / 150) * 150
    offgrid = np.linalg.norm(points - nodes, axis=1) > 1e-7
    node_distance = np.linalg.norm(points - nodes, axis=1) / 5
    node_to_next = np.linalg.norm(nodes[:, None, :] - points[None, :, :], axis=2) / 5

    def components(o):
        o = np.array(o, dtype=int)
        m = len(o)
        scan = o >= source_count
        seen = np.logical_or.accumulate(vis[o], axis=0)
        before = np.r_[np.zeros((1, len(ids)), bool), seen[:-1]]
        fees = float(6 * np.sum(blank + ((~before)[scan] @ weights)))
        edges = dd[o[:-1] + 1, o[1:] + 1]
        travel = float(dd[0, o[0] + 1] + edges.sum())
        insert = np.r_[dg[o[:-1]] + dg[o[1:]] - edges[:, None], dg[o[-1:]]]
        suffix = np.minimum.accumulate(insert[::-1], axis=0)[::-1]
        first = np.argmax(vis[o], axis=0)
        assert np.all(
            seen[-1]
        ), "Certified future scans must see every retained hypothetical source"
        future = float(weights @ suffix[first, np.arange(len(ids))])
        if getattr(bot, "regional_anchor", False):
            mass = np.bincount(first, weights=weights, minlength=len(o))
            travel_penalty = np.r_[
                node_distance[o[:-1]] + node_to_next[o[:-1], o[1:]] - edges,
                node_distance[o[-1:]],
            ]
            # First-hit source anchors pay6s each; the shared trip to the
            # nearest lattice vertex is priced once with a Poisson occupancy
            # approximation. Probe fees common to all orders are omitted.
            anchor = float(
                6 * (weights @ offgrid[o[first]])
                + np.sum((1 - np.exp(-mass)) * np.maximum(0, travel_penalty))
            )
            return travel, fees, future, anchor
        return travel, fees, future

    def score(o):
        key = tuple(o)
        if key not in cache:
            cache[key] = sum(components(o))
        return cache[key]

    previous = [keys.index(k) for k in bot.route_keys if k in keys]
    geometric, _, length = optimize_route(bot.position, points.tolist(), previous)
    seeds = [geometric]
    # Build the patrol first, then place each source visit where its neighborhood
    # has enough discovery information to avoid a second excursion.
    scanorder, _, _ = optimize_route(bot.position, points[source_count:].tolist())
    scanorders = [scanorder]
    for scanning in scanorders:
        for sourceorder in (
            list(range(source_count)),
            list(reversed(range(source_count))),
        ):
            order = [source_count + i for i in scanning]
            for c in sourceorder:
                order = min(
                    (order[:j] + [c] + order[j:] for j in range(len(order) + 1)),
                    key=score,
                )
            seeds.append(order)
    if len(previous) == n:
        seeds.append(previous)
    best = min(seeds, key=score)
    value = score(best)
    for _ in range(3):
        candidate = None
        cv = value
        for i in range(n - 1):
            for j in range(i + 1, n):
                trial = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                v = score(trial)
                if v < cv - 1e-8:
                    candidate = trial
                    cv = v
        for i in range(n):
            rest = best[:i] + best[i + 1 :]
            for j in range(n):
                if i == j:
                    continue
                trial = rest[:j] + [best[i]] + rest[j:]
                v = score(trial)
                if v < cv - 1e-8:
                    candidate = trial
                    cv = v
        if candidate is None:
            break
        best = candidate
        value = cv
    assert sorted(best) == list(range(n))
    bot.route_keys = [keys[i] for i in best]
    bot.regional_diagnostics.append(
        dict(
            action_count=len(bot.log),
            expected_unseen=h["expected"],
            objective_before_s=score(geometric),
            objective_after_s=value,
            geometry_before_m=length,
            components_before=components(geometric),
            components_after=components(best),
            selected=tasks[best[0]],
            changed=tasks[best[0]] != baseline,
            evaluations=len(cache),
        )
    )
    return tasks[best[0]]
