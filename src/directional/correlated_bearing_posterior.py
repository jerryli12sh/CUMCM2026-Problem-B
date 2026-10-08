"""Soft source-location inference from already paid bearings, never a hard proof.

The private smoothstep field is approximated by a Gaussian over its grid
coefficients (variance1/3) plus bearing quantization variance. It is an explicit
planning approximation, not an assertion about official hidden noise. Only
ordinary bearing coordinates/values and the existing hard polygon are inputs.
"""

import math
import numpy as np
from gradient import project_polygon


def features(points, offset=(0.0, 0.0)):
    rows = []
    keys = []
    for x, y in points:
        xx, yy = (x + offset[0]) / 150, (y + offset[1]) / 150
        ix, iy = math.floor(xx), math.floor(yy)
        u, v = xx - ix, yy - iy
        a, b = u * u * (3 - 2 * u), v * v * (3 - 2 * v)
        row = {
            (ix, iy): (1 - a) * (1 - b),
            (ix + 1, iy): a * (1 - b),
            (ix, iy + 1): (1 - a) * b,
            (ix + 1, iy + 1): a * b,
        }
        rows.append(row)
        keys.extend(row)
    keys = sorted(set(keys))
    return np.array([[row.get(k, 0.0) for k in keys] for row in rows])


def inside_polygon(points, poly):
    inside = np.ones(len(points), dtype=bool)
    for a, b in zip(poly, poly[1:] + poly[:1]):
        inside &= (
            (b[0] - a[0]) * (points[:, 1] - a[1])
            - (b[1] - a[1]) * (points[:, 0] - a[0])
        ) >= -1e-6
    return inside


def infer(belief, proposal, minimum_gain=0.03, use_mean=False):
    bearings = list(belief.bearings[-6:])
    if len(bearings) < 2 or len(belief.poly) < 3:
        return tuple(proposal), dict(used=False, reason="insufficient_bearings")
    points = np.asarray([q for q, _ in bearings])
    angles = np.degrees([beta for _, beta in bearings])
    grid = [
        i
        for i, q in enumerate(points)
        if all(abs(v / 150 - round(v / 150)) < 1e-7 for v in q)
    ]
    anchor_index = grid[-1] if grid else 0
    anchor = points[anchor_index]
    beta = math.radians(angles[anchor_index])
    radii = np.arange(2.5, 1500.01, 5.0)
    offsets = np.linspace(-1.0, 1.0, 81) * math.pi / 180
    r, t = np.meshgrid(radii, beta + offsets, indexing="ij")
    r = r.ravel()
    t = t.ravel()
    particles = anchor + np.column_stack((r * np.cos(t), r * np.sin(t)))
    keep = inside_polygon(particles, list(belief.poly)) & (
        np.linalg.norm(particles, axis=1) <= 1800 + 1e-6
    )
    for q in belief.failed:
        keep &= np.linalg.norm(particles - np.asarray(q), axis=1) > 20
    particles = particles[keep]
    prior = r[keep]
    if not len(particles):
        return tuple(proposal), dict(used=False, reason="finite_posterior_grid_empty")
    delta = particles[:, None, :] - points[None, :, :]
    predicted = np.degrees(np.arctan2(delta[:, :, 1], delta[:, :, 0]))
    residual = (angles - predicted + 180) % 360 - 180
    keep = np.max(np.abs(residual), axis=1) <= 1 + 1e-6
    particles = particles[keep]
    prior = prior[keep]
    residual = residual[keep]
    delta = delta[keep]
    if not len(particles):
        return tuple(proposal), dict(used=False, reason="bounded_bearing_support_empty")
    weight_matrix = features(points)
    covariance = weight_matrix @ weight_matrix.T / 3 + np.eye(len(points)) * (
        0.01**2 / 12
    )
    vals, vecs = np.linalg.eigh(covariance)
    vals = np.maximum(vals, 1e-10)
    whitened = (residual @ vecs) / np.sqrt(vals)
    loglike = -0.5 * np.sum(whitened**2, axis=1)
    # Uniform source area in polar quadrature, and an explicitly assumed
    # uniform receive radius1000..1500 conditional on all positive readings.
    max_range = np.max(np.linalg.norm(delta, axis=2), axis=1)
    survival = np.minimum(1.0, np.maximum(0.0, (1500 - max_range) / 500))
    logweight = loglike + np.log(prior) + np.log(np.maximum(survival, 1e-300))
    weights = np.exp(logweight - logweight.max())
    weights /= weights.sum()
    mean = weights @ particles
    centered = particles - mean
    spatial_cov = (centered * weights[:, None]).T @ centered
    if use_mean:
        candidates = [tuple(proposal), tuple(project_polygon(tuple(mean), belief.poly))]
    else:
        ev, evec = np.linalg.eigh(spatial_cov)
        axes = evec * np.sqrt(np.maximum(ev, 0))
        candidates = [tuple(proposal), tuple(mean)]
        for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for b in (-1.0, -0.5, 0.0, 0.5, 1.0):
                candidates.append(tuple(mean + axes @ np.array((a, b))))
        # Include fine-grid high-posterior points to handle skewed truncation.
        top = np.argsort(weights)[-24:]
        candidates.extend(tuple(p) for p in particles[top])
        candidates = list(
            dict.fromkeys(tuple(project_polygon(q, belief.poly)) for q in candidates)
        )
    masses = [
        float(weights @ (np.linalg.norm(particles - np.asarray(q), axis=1) <= 20))
        for q in candidates
    ]
    best = int(np.argmax(masses))
    baseline_mass = masses[0]
    used = best != 0 and masses[best] - baseline_mass >= minimum_gain
    chosen = candidates[best] if used else tuple(proposal)
    diagnostic = dict(
        used=used,
        bearings=[(tuple(q), float(b)) for q, b in bearings],
        anchor_index=anchor_index,
        ordinary_input_polygon=belief.poly,
        original_proposal=proposal,
        posterior_mean=mean.tolist(),
        posterior_covariance=spatial_cov.tolist(),
        noise_covariance=covariance.tolist(),
        grid_particles=len(particles),
        effective_particles=float(1 / (weights @ weights)),
        baseline_success_mass=baseline_mass,
        candidate_success_mass=masses[best],
        chosen=chosen,
        displacement_m=math.dist(chosen, proposal),
        approximation="Gaussian smoothstep-coefficient prior; finite position quadrature; no hard-set tightening",
    )
    return chosen, diagnostic
