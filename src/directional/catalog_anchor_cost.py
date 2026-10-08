"""Price first-discovery lattice anchoring on a candidate initial route."""

import math
import numpy as np
from predictive_route import survival_table
from rollout_type_prior import condition


def cost(bot, known, points, order):
    survival_table(bot, [])
    h = condition(bot)
    if h["expected"] <= 0:
        return 0.0
    xy, radius, normals = bot.particle_cache
    n0 = len(xy)
    observed = sum(b.known or b.cleared for b in bot.beliefs.values())
    eo = h["expected_omni"]
    ed = h["expected_directional"]
    weights = np.r_[
        np.full(n0, eo / max(1, h["alive"][:n0].sum())),
        np.full(n0, ed / max(1, h["alive"][n0:].sum())),
    ]
    alive = h["alive"].copy()
    tasks = known + points[1:]
    total = 0.0
    for at, i in enumerate(order):
        if i < len(known):
            continue
        p = tasks[i]
        delta = np.asarray(p) - xy
        om = np.sum(delta * delta, axis=1) <= radius * radius
        vis = np.r_[om, om & (np.sum(delta * normals, axis=1) >= 0)]
        mass = float(weights @ (alive & vis))
        alive &= ~vis
        node = tuple(150.0 * round(v / 150) for v in p)
        if math.dist(node, p) < 1e-8:
            continue
        # All new channel anchors pay their own measurement and switching;
        # travel from the scan to its shared nearest node is paid once.
        detour = math.dist(p, node)
        if at + 1 < len(order):
            end = tasks[order[at + 1]]
            detour += math.dist(node, end) - math.dist(p, end)
        total += 6 * mass + (1 - math.exp(-mass)) * max(0, detour) / 5
    return total
