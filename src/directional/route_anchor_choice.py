"""Choose a probe vertex using ordinary first bearings and future travel.

Single-bearing particles integrate the uniform lattice field and shared
radius/orientation receipts. They are a soft ranking tool. An actual no-signal
reply still invokes the ordinary alternate-corner fallback; absence proofs and
conservative source envelopes remain unchanged.
"""

import math
from types import SimpleNamespace
import numpy as np
from scipy.stats import qmc
from correlated_bearing_posterior import features, inside_polygon
from signal_history_likelihood import likelihood
from uniform_field_posterior import quantize
from route_optimizer import optimize_route

_BANK = {}


def posterior(belief):
    if len(belief.bearings) != 1:
        return None
    q, beta = belief.bearings[0]
    W = features([q])[0]
    dim = len(W) + 2
    if dim not in _BANK:
        _BANK[dim] = qmc.Sobol(dim, scramble=False).random_base2(12)
    z = _BANK[dim]
    error = (2 * z[:, :-2] - 1) @ W
    theta = (
        math.degrees(beta) - np.clip(error, -0.995, 0.995) + (z[:, -2] - 0.5) * 0.01
    ) % 360
    radius = np.sqrt(25 + (1500**2 - 25) * z[:, -1])
    angle = np.radians(theta)
    particles = np.asarray(q) + radius[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    keep = (np.linalg.norm(particles, axis=1) <= 1770) & inside_polygon(
        particles, list(belief.poly)
    )
    keep &= quantize(theta, error) == round(math.degrees(beta) * 100) % 36000
    particles = particles[keep]
    if len(particles) < 64:
        return None
    values = likelihood(belief, particles)
    if values.sum() <= 0:
        return None
    weights = values / values.sum()
    ess = float(1 / (weights @ weights))
    if ess < 64:
        return None
    return (
        particles,
        weights,
        values,
        dict(particles=len(particles), effective_particles=ess),
    )


def visible_probability(belief, value, node):
    particles, weights, base, _ = value
    updated = SimpleNamespace(
        positive=list(belief.positive) + [node],
        negative=list(belief.negative),
        directional_prior=getattr(belief, "directional_prior", 0.5),
    )
    extra = likelihood(updated, particles)
    probability = float(
        weights @ np.divide(extra, base, out=np.zeros_like(extra), where=base > 0)
    )
    assert -0.000001 <= probability <= 1.000001
    return min(1.0, max(0.0, probability))


def choose(bot, q, prior):
    nearest = tuple(float(150 * round(v / 150)) for v in q)
    if math.dist(nearest, q) < 1e-7:
        return nearest
    fresh = [
        c
        for c, b in bot.beliefs.items()
        if c not in prior and b.known and not b.cleared and b.circle()[1] > 100
    ]
    if not fresh:
        return nearest
    values = {c: posterior(bot.beliefs[c]) for c in fresh}
    if any(v is None for v in values.values()):
        return nearest
    tasks = []
    task_distributions = []
    for c, b in bot.beliefs.items():
        if not b.known or b.cleared:
            continue
        if c in values:
            tasks.append(tuple(values[c][1] @ values[c][0]))
            task_distributions.append(values[c][:2])
        else:
            tasks.append(bot.source_target(c))
            task_distributions.append(None)
    scanning = [p for p in bot.pending_stations if math.dist(p, q) > 1e-6]
    tasks += scanning
    task_distributions += [None] * len(scanning)
    if not tasks:
        return nearest
    order, _, _ = optimize_route(q, tasks)
    end = tasks[order[0]]
    distribution = (
        task_distributions[order[0]]
        if bot.route_anchor in ("expected", "wide_expected")
        else None
    )
    x = 150 * math.floor(q[0] / 150)
    y = 150 * math.floor(q[1] / 150)
    nodes = [nearest] + [
        (float(x + dx), float(y + dy))
        for dx in (0, 150)
        for dy in (0, 150)
        if (float(x + dx), float(y + dy)) != nearest
    ]
    wide = bot.route_anchor in ("wide", "wide_expected")
    if wide:
        extras = [
            (nearest[0] + 150 * dx, nearest[1] + 150 * dy)
            for dx in range(-3, 4)
            for dy in range(-3, 4)
        ]
        nodes += sorted(
            (v for v in extras if v not in nodes and math.dist(q, v) <= 450.0),
            key=lambda v: (math.dist(q, v), v),
        )
    records = []
    for node in nodes:
        outgoing = (
            math.dist(node, end)
            if distribution is None
            else float(
                distribution[1]
                @ np.linalg.norm(distribution[0] - np.asarray(node), axis=1)
            )
        )
        travel = (math.dist(bot.position, node) + outgoing) / 5
        if wide and records and travel > records[0]["score"] - 2.0:
            records.append(
                dict(
                    node=node,
                    visibility=None,
                    travel_s=travel,
                    score=None,
                    pruned_by_geometric_lower_bound=True,
                )
            )
            continue
        probabilities = [
            visible_probability(bot.beliefs[c], values[c], node) for c in fresh
        ]
        score = travel + 50 * sum(1 - p for p in probabilities)
        records.append(
            dict(node=node, visibility=probabilities, travel_s=travel, score=score)
        )
    allowed = [
        i
        for i, r in enumerate(records)
        if i == 0 or (r["visibility"] is not None and min(r["visibility"]) >= 0.97)
    ]
    chosen = min(allowed, key=lambda i: records[i]["score"])
    if records[0]["score"] - records[chosen]["score"] < 2.0:
        chosen = 0
    bot.route_anchor_diagnostics.append(
        dict(
            station=q,
            fresh=fresh,
            expected_next=end,
            nearest=nearest,
            chosen=nodes[chosen],
            changed=chosen != 0,
            candidates=records,
            predicted_saving_s=records[0]["score"] - records[chosen]["score"],
            outgoing_integrated_over_source_distribution=distribution is not None,
        )
    )
    return nodes[chosen]
