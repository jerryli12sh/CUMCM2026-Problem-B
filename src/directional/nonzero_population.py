"""D|N uniform1..N, conditioned on ordinary first-discovery receipts.

UniformN10..16 is the retained soft prior, not hidden actualN. Marked-source
quadrature is nonnegative. Target-source type odds leave its own receipt
factor out so the source likelihood can use that history without duplicating
it. Other sources contribute their coarse discovery-time evidence only.
"""

import math
import numpy as np
from nonzero_directional_prior_kernel import integrate_categories


def categories(bot):
    xy, ranges, normals = bot.particle_cache
    npart = len(xy)
    alive = np.ones(2 * npart, bool)
    qo = qd = 1.0
    records = []
    count = 0
    membership = {}
    for point in bot.discovery_stations:
        delta = np.asarray(point) - xy
        om = (delta * delta).sum(axis=1) <= ranges * ranges
        vis = np.r_[om, om & ((delta * normals).sum(axis=1) >= 0)]
        alive &= ~vis
        no, nd = alive[:npart].mean(), alive[npart:].mean()
        found = [
            c
            for c, b in bot.beliefs.items()
            if b.positive and math.dist(b.positive[0], point) < 1e-7
        ]
        if found:
            ao, ad = max(0.0, float(qo - no)), max(0.0, float(qd - nd))
            # Only a resolution fallback in the soft particle model. It never
            # changes absence proofs or a conservative source envelope.
            if ao == 0 and ad == 0:
                ao = ad = 1e-12
            index = len(records) + 1
            records.append((len(found), ao, ad))
            for ch in found:
                membership[ch] = index
        count += len(found)
        qo, qd = float(no), float(nd)
    assert count == sum(b.known or b.cleared for b in bot.beliefs.values())
    return (
        np.array([qo] + [r[1] for r in records]),
        np.array([qd] + [r[2] for r in records]),
        [r[0] for r in records],
        count,
        alive,
        membership,
    )


def condition(bot):
    a, b, counts, m, alive, membership = categories(bot)
    ns = np.arange(max(10, m), 17)
    masses = []
    omnis = []
    dirs = []
    for n in ns:
        k = int(n) - m
        r = integrate_categories(a, b, [k] + counts)
        factor = math.factorial(int(n)) / math.factorial(k)
        masses.append(factor * r["likelihood"])
        omnis.append(factor * r["unseen_omni_numerator"])
        dirs.append(factor * r["unseen_directional_numerator"])
    total = sum(masses)
    assert total > 0, "Finite particle discovery model has no supportedN"
    eo, ed = sum(omnis) / total, sum(dirs) / total
    assert eo >= 0 and ed >= 0 and eo + ed <= 16 - m + 1e-8
    return dict(
        ns=ns,
        prob_N=np.array(masses) / total,
        expected=eo + ed,
        expected_omni=eo,
        expected_directional=ed,
        qo=float(a[0]),
        qd=float(b[0]),
        alive=alive,
        rule="D1..N",
    )


def source_type_priors(bot):
    from predictive_route import survival_table

    survival_table(bot, [])
    a, b, counts, m, alive, membership = categories(bot)
    ns = range(max(10, m), 17)
    result = {}
    by_category = {}
    for index in sorted(set(membership.values())):
        other = counts.copy()
        other[index - 1] -= 1
        mass = omni = 0.0
        # Reorder so category0 is the tagged, present source. Its own
        # discovery/measurement likelihood is deliberately replaced by1.
        aa = np.r_[1.0, a]
        bb = np.r_[1.0, b]
        for n in ns:
            k = n - m
            r = integrate_categories(aa, bb, [1, k] + other)
            factor = math.factorial(n) / math.factorial(k)
            mass += factor * r["likelihood"]
            omni += factor * r["unseen_omni_numerator"]
        assert mass > 0
        by_category[index] = min(1.0, max(0.0, 1 - omni / mass))
    for ch, index in membership.items():
        value = by_category[index]
        belief = bot.beliefs[ch]
        if value != getattr(belief, "directional_prior", 0.5):
            belief.directional_prior = value
            if hasattr(bot, "uniform_cache"):
                bot.uniform_cache.pop(ch, None)
        result[ch] = value
    if hasattr(bot, "nonzero_prior_diagnostics"):
        bot.nonzero_prior_diagnostics.append(
            dict(
                discovery_stations=len(bot.discovery_stations),
                known=m,
                source_priors=result,
            )
        )
    return result
