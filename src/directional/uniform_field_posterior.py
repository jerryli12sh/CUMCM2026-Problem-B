"""Soft importance integration under independent uniform lattice errors.

Two quantized bearings are inverted geometrically. Remaining ordinary bearings
condition the same sampled lattice field, using the actual rounding/clamping
rule. The change-of-variables weight is1/|det d(bearings)/d(position)|.
No real scene, random seed, source count, or future reply is accessed.
"""

import math
import numpy as np
from scipy.stats import qmc
from correlated_bearing_posterior import (
    features,
    inside_polygon,
    infer as gaussian_infer,
)
from gradient import project_polygon
from clear_neighborhood import best_clear_point
from signal_history_likelihood import likelihood

_QUADRATURE = {}


def quantize(theta, error):
    raw = (theta + error) * 100
    integer = np.where(raw >= 0, np.floor(raw + 0.5), np.ceil(raw - 0.5))
    integer = np.maximum(
        np.ceil((theta - 1) * 100), np.minimum(np.floor((theta + 1) * 100), integer)
    )
    return np.remainder(integer, 36000)


def integrate(belief, proposal):
    bearings = list(belief.bearings[-6:])
    if len(bearings) < 2:
        return None
    points = np.array([p for p, _ in bearings])
    angles = np.degrees([b for _, b in bearings]) % 360
    anchors = [
        i
        for i, p in enumerate(points)
        if all(abs(v / 150 - round(v / 150)) < 1e-8 for v in p)
    ]
    if not anchors:
        return None
    ai = anchors[-1]
    near = [
        i
        for i, p in enumerate(points)
        if i != ai and 0.5 < np.linalg.norm(p - points[ai]) < 21
    ]
    if not near:
        return None
    bi = min(near, key=lambda i: np.linalg.norm(points[i] - points[ai]))
    W = features(points)
    dim = W.shape[1] + 2
    if dim not in _QUADRATURE:
        _QUADRATURE[dim] = qmc.Sobol(dim, scramble=False).random_base2(16)
    z = _QUADRATURE[dim]
    errors = (2 * z[:, :-2] - 1) @ W.T
    # For fixed error e, the preimage of a hundredth-degree report is a
    # .01-degree interval centered at report-clip(e,-.995,.995). This accounts
    # for the final +/-1-degree integer clamp as well as ordinary rounding;
    # the exact forward test below handles endpoint/tie details.
    theta0 = np.radians(
        angles[ai] - np.clip(errors[:, ai], -0.995, 0.995) + (z[:, -2] - 0.5) * 0.01
    )
    theta1 = np.radians(
        angles[bi] - np.clip(errors[:, bi], -0.995, 0.995) + (z[:, -1] - 0.5) * 0.01
    )
    u = np.c_[np.cos(theta0), np.sin(theta0)]
    v = np.c_[np.cos(theta1), np.sin(theta1)]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    step = points[bi] - points[ai]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (step[0] * v[:, 1] - step[1] * v[:, 0]) / cross
    safe = np.isfinite(r) & (r > 5) & (r < 1500.001)
    r = r[safe]
    u = u[safe]
    v = v[safe]
    cross = cross[safe]
    errors = errors[safe]
    particles = points[ai] + r[:, None] * u
    d = particles[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(d, axis=2)
    physical = (
        np.all(dist > 5, axis=1)
        & np.all(dist <= 1500, axis=1)
        & (np.linalg.norm(particles, axis=1) <= 1770)
    )
    physical &= np.sum(d[:, bi, :] * v, axis=1) > 0
    theta = np.degrees(np.arctan2(d[:, :, 1], d[:, :, 0])) % 360
    physical &= np.all(
        quantize(theta, errors) == np.rint(angles * 100)[None, :], axis=1
    )
    physical &= inside_polygon(particles, list(belief.poly))
    for failed in belief.failed:
        physical &= np.linalg.norm(particles - np.array(failed), axis=1) > 20
    particles = particles[physical]
    dist = dist[physical]
    cross = cross[physical]
    r = r[physical]
    if len(particles) < 64:
        return None
    survival = likelihood(belief, particles)
    jac = np.abs(cross) / (r * dist[:, bi])
    weights = survival / np.maximum(jac, 1e-18)
    if not weights.sum():
        return None
    weights /= weights.sum()
    ess = float(1 / (weights @ weights))
    if ess < 64:
        return None
    return (
        particles,
        weights,
        dict(
            particles=len(particles),
            effective_particles=ess,
            anchor_index=ai,
            probe_index=bi,
            assumption="uniform node errors; quantization-conditioned Sobol integration; uniform area/radius, half type prior, shared orientation/radius conditioned on all signal receipts",
        ),
    )


def infer(belief, proposal):
    value = integrate(belief, proposal)
    if value is None:
        point, diag = gaussian_infer(belief, proposal, minimum_gain=0.03, use_mean=True)
        return point, dict(diag, uniform_fallback=True), None
    particles, weights, diag = value
    mean = project_polygon(tuple(weights @ particles), belief.poly)
    masses = [
        float(weights @ (np.linalg.norm(particles - np.asarray(q), axis=1) <= 20))
        for q in (proposal, mean)
    ]
    used = masses[1] - masses[0] >= 0.03
    chosen = mean if used else proposal
    diag.update(
        used=used,
        baseline_success_mass=masses[0],
        candidate_success_mass=masses[1],
        chosen=chosen,
        displacement_m=math.dist(chosen, proposal),
    )
    return tuple(chosen), diag, value


def clearance(bot, ch, penalty):
    b = bot.beliefs[ch]
    proposal = bot.source_target(ch)
    cached = bot.uniform_cache.get(ch)
    value = cached[1] if cached and cached[0] == b.version else integrate(b, proposal)
    if value is None:
        return
    particles, weights, diag = value
    end = None
    for kind, key in bot.route_keys:
        if kind == "source" and key != ch and not bot.beliefs[key].cleared:
            end = bot.source_target(key)
            break
        if kind == "scan" and key in bot.pending_stations:
            end = key
            break
    candidates = [proposal]
    for center in (proposal, tuple(weights @ particles)):
        candidates.append(tuple(project_polygon(center, b.poly)))
        for radius in (3.0, 6.0, 9.0, 12.0, 15.0):
            candidates.append(
                best_clear_point(center, radius, bot.position, end, b.poly)
            )
    candidates = list(dict.fromkeys(candidates))
    mass = np.array(
        [
            float(weights @ (np.linalg.norm(particles - np.asarray(q), axis=1) <= 20))
            for q in candidates
        ]
    )
    movement = np.array(
        [
            (math.dist(bot.position, q) + (math.dist(q, end) if end is not None else 0))
            / 5
            for q in candidates
        ]
    )
    cost = movement + penalty * (1 - mass)
    allow = (mass >= mass[0] - 0.05) & (mass >= min(0.65, mass[0]))
    cost[~allow] = math.inf
    chosen = int(np.argmin(cost))
    if cost[chosen] < cost[0] - 0.1:
        bot.gradient_targets[ch] = candidates[chosen]
    else:
        chosen = 0
    bot.clear_utility_diagnostics.append(
        dict(
            channel=ch,
            position=bot.position,
            next=end,
            original=proposal,
            chosen=candidates[chosen],
            shift_m=math.dist(proposal, candidates[chosen]),
            success_before=float(mass[0]),
            success_after=float(mass[chosen]),
            predicted_net_s=float(cost[0] - cost[chosen]),
            penalty_s=penalty,
            particles=diag["particles"],
            effective_particles=diag["effective_particles"],
        )
    )
