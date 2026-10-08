"""Optional optical trials on the already-planned incoming segment.

If every early trial fails, the original target is still attempted at its exact
coordinate. Thus early failures add service fees but no approach-path length.
All attempts use real success receipts and retain the conservative fallback.
"""

import math, itertools
import numpy as np
from uniform_field_posterior import integrate


def attempt(bot, ch):
    b = bot.beliefs[ch]
    target = bot.source_target(ch)
    start = bot.position
    length = math.dist(start, target)
    if length < 40:
        return False, False, target
    direction = ((target[0] - start[0]) / length, (target[1] - start[1]) / length)
    if bot.preclear == "fixed":
        selected = [tuple(target[i] - 16 * direction[i] for i in (0, 1))]
        diag = dict(mode="fixed16", selected=selected, predicted_gain_s=None)
    else:
        cached = bot.uniform_cache.get(ch)
        posterior = (
            cached[1] if cached and cached[0] == b.version else integrate(b, target)
        )
        if posterior is None:
            return False, False, target
        particles, weight, pdiag = posterior
        end = None
        for kind, key in bot.route_keys:
            if kind == "source" and key != ch and not bot.beliefs[key].cleared:
                end = bot.source_target(key)
                break
            if kind == "scan" and key in bot.pending_stations:
                end = key
                break
        distances = [d for d in (60.0, 40.0, 32.0, 24.0, 16.0, 8.0) if d < 0.8 * length]
        points = [
            tuple(target[i] - d * direction[i] for i in (0, 1)) for d in distances
        ]
        success = np.array(
            [np.linalg.norm(particles - np.array(q), axis=1) <= 20 for q in points]
        )
        original = np.linalg.norm(particles - np.array(target), axis=1) <= 20
        saved = np.array(
            [
                (d + (math.dist(target, end) - math.dist(q, end) if end else 0)) / 5
                for d, q in zip(distances, points)
            ]
        )
        bestgain = 0.5
        chosen = None
        details = None
        # At most two probes on the incoming segment; conditioning on an early
        # failure is accounted for by the disjoint first-success probabilities.
        for count in (1, 2):
            for indices in itertools.combinations(range(len(points)), count):
                covered = np.zeros(len(weight), bool)
                gain = 0.0
                first = []
                failure_fee = 0.0
                for i in indices:
                    newly = success[i] & ~covered
                    prob = float(weight @ newly)
                    first.append(prob)
                    gain += prob * saved[i]
                    covered |= success[i]
                    failure_fee += 3 * float(weight @ (~covered))
                exclusive = float(weight @ (covered & ~original))
                gain += exclusive * (50.0 - 2.0) - failure_fee
                if gain > bestgain:
                    bestgain = gain
                    chosen = indices
                    details = dict(
                        first_success_prob=first,
                        exclusive_success=exclusive,
                        expected_extra_failure_fees=failure_fee,
                    )
        if chosen is None:
            return False, False, target
        selected = [points[i] for i in chosen]
        diag = dict(
            mode="posterior_staged",
            selected=selected,
            predicted_gain_s=bestgain,
            **details,
            particles=pdiag["particles"],
            effective_particles=pdiag["effective_particles"],
            fallback_cost_assumption_s=50.0
        )
    diag.update(channel=ch, start=start, original=target, results=[])
    bot.preclear_diagnostics.append(diag)
    for p in selected:
        success = bot.clear(p, ch, safe=False)
        diag["results"].append(success)
        if success:
            return True, True, target
    return True, False, target
