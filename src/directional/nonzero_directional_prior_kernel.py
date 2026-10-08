"""Nonnegative quadrature for a directional-source count uniform on 1..N."""

import numpy as np


def integrate_categories(a, b, counts):
    a, b = np.asarray(a, float), np.asarray(b, float)
    counts = np.asarray(counts, int)
    assert a.shape == b.shape == counts.shape and len(a) > 0
    n = int(counts.sum())
    assert 1 <= n <= 16 and np.all(counts >= 0)
    assert (
        np.all(np.isfinite(a))
        and np.all(np.isfinite(b))
        and np.all(a >= 0)
        and np.all(b >= 0)
    )
    nodes, w = np.polynomial.legendre.leggauss(9)
    eta = (nodes + 1) / 2
    w = w / 2
    p = a[:, None] * (1 - eta) + b[:, None] * eta
    density = np.zeros(9)
    omni_density = np.zeros(9)
    for j in np.flatnonzero(counts):
        exponents = counts.copy()
        exponents[j] -= 1
        factor = counts[j] * b[j] / n
        density += factor * np.prod(p ** exponents[:, None], axis=0)
        if exponents[0] > 0:
            multiplicity = exponents[0]
            exponents[0] -= 1
            omni_density += (
                factor
                * multiplicity
                * a[0]
                * (1 - eta)
                * np.prod(p ** exponents[:, None], axis=0)
            )
    evidence = float(w @ density)
    omni_numerator = float(w @ omni_density)
    directional_numerator = counts[0] * evidence - omni_numerator
    assert (
        evidence >= 0
        and omni_numerator >= 0
        and directional_numerator >= -1e-14 * max(evidence, 1e-300)
    )
    return dict(
        likelihood=evidence,
        unseen_omni_numerator=omni_numerator,
        unseen_directional_numerator=max(0.0, float(directional_numerator)),
    )
