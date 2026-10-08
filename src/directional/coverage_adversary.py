"""Finite geometric witnesses only; never a certificate. Extracted from existing geometry research."""

import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def base_positions(domain=1.8):
    a = np.arange(240) * 2 * np.pi / 240
    boundary = 1.8 * np.column_stack((np.cos(a), np.sin(a)))
    inner = []
    for r in np.arange(0.15, 1.8, 0.15):
        a = (np.arange(48) + 0.37) * 2 * np.pi / 48
        inner.extend(r * np.column_stack((np.cos(a), np.sin(a))))
    return np.vstack((np.zeros((1, 2)), inner, boundary)) * (domain / 1.8)


def violations(p, g, u, radius=1.0):
    d = p[None, :, :] - g[:, None, :]
    return np.maximum(
        np.linalg.norm(d, axis=2) - radius, -np.einsum("msd,md->ms", d, u)
    )


def intersections(a, ra, b, rb):
    d = np.linalg.norm(b - a)
    if d < 1e-10 or d > ra + rb or d < abs(ra - rb):
        return []
    t = (ra * ra - rb * rb + d * d) / (2 * d)
    h = math.sqrt(max(0.0, ra * ra - t * t))
    v = (b - a) / d
    n = np.array((-v[1], v[0]))
    c = a + t * v
    return [c + h * n, c - h * n]


def refine_holes(p, g, u, domain=1.8, starts=24):
    """Move source position AND heading to maximize a valid violation."""
    outg = []
    outu = []
    seen = set()

    def obj(y, tau):
        x = y[:2]
        u = np.array((np.cos(y[2]), np.sin(y[2])))
        up = np.array((-u[1], u[0]))
        d = p - x
        dn = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        a = dn - 1.0
        b = -d @ u
        v = tau * np.logaddexp(a / tau, b / tau)
        wa = expit((a - b) / tau)
        z = -v / tau
        normalizer = logsumexp(z)
        ws = np.exp(z - normalizer)
        value = -tau * normalizer
        gx = wa[:, None] * (-d / dn[:, None]) + (1 - wa)[:, None] * u
        gt = -(1 - wa) * (d @ up)
        grad = np.r_[ws @ gx, ws @ gt]
        return -float(value), -grad

    for x, d in zip(g, u):
        key = tuple(np.round(x / 0.08).astype(int))
        if key in seen:
            continue
        seen.add(key)
        y = np.r_[x, np.arctan2(d[1], d[0])]
        for tau in (0.001, 0.00005):
            res = minimize(
                obj,
                y,
                args=(tau,),
                jac=True,
                method="SLSQP",
                bounds=[
                    (-domain, domain),
                    (-domain, domain),
                    (y[2] - np.pi, y[2] + np.pi),
                ],
                constraints={
                    "type": "ineq",
                    "fun": lambda y: domain * domain - y[:2] @ y[:2],
                    "jac": lambda y: np.r_[-2 * y[:2], 0.0],
                },
                options=dict(maxiter=45, ftol=1e-11),
            )
            y = res.x
        x = y[:2]
        r = np.linalg.norm(x)
        if r > domain:
            x = x * (domain / r)
        outg.append(x)
        outu.append((np.cos(y[2]), np.sin(y[2])))
        if len(outg) >= starts:
            break
    return np.asarray(outg).reshape(-1, 2), np.asarray(outu).reshape(-1, 2)


def adversary(p, extra=None, dense=False, domain=1.8, refine=False):
    """Actual geometric holes and gap normals; this is not a proof oracle."""
    positions = [base_positions(domain)]
    a = np.arange(1440 if dense else 720) * 2 * np.pi / (1440 if dense else 720)
    positions.append(domain * np.column_stack((np.cos(a), np.sin(a))))
    a = np.arange(12) * np.pi / 6
    directions = np.column_stack((np.cos(a), np.sin(a)))
    positions.append(p)
    for eps in (1e-6, 0.00001, 0.0001, 0.001, 0.01):
        positions.append((p[:, None, :] + eps * directions[None, :, :]).reshape(-1, 2))
    centers = []
    for j in range(len(p)):
        centers.extend(intersections(p[j], 1.0, np.zeros(2), domain))
        for k in range(j):
            centers.extend(intersections(p[j], 1.0, p[k], 1.0))
    if centers:
        c = np.array(centers)
        positions.append(c)
        a = np.arange(12) * np.pi / 6
        ds = np.column_stack((np.cos(a), np.sin(a)))
        for eps in (
            (1e-7, 1e-5, 0.0002, 0.002, 0.01) if dense else (1e-6, 0.0002, 0.003)
        ):
            positions.append((c[:, None, :] + eps * ds[None, :, :]).reshape(-1, 2))
    if extra is not None:
        positions.append(np.asarray(extra).reshape(-1, 2))
    g = np.vstack(positions)
    g = g[np.linalg.norm(g, axis=1) <= domain + 1e-12]
    g = np.unique(np.round(g, 11), axis=0)
    d = p[None, :, :] - g[:, None, :]
    dist = np.linalg.norm(d, axis=2)
    angles = np.where(
        dist <= 1.0 + 1e-11, np.arctan2(d[:, :, 1], d[:, :, 0]) % (2 * np.pi), 10.0
    )
    angles.sort(axis=1)
    cnt = (dist <= 1.0 + 1e-11).sum(axis=1)
    gaps = np.diff(angles, axis=1)
    valid = np.arange(len(p) - 1)[None, :] < cnt[:, None] - 1
    gaps = np.where(valid, gaps, -1.0)
    wrap = angles[:, 0] + 2 * np.pi - angles[np.arange(len(g)), np.maximum(cnt - 1, 0)]
    gaps = np.column_stack((gaps, np.where(cnt > 0, wrap, 2 * np.pi)))
    idx = np.argmax(gaps, axis=1)
    gap = gaps[np.arange(len(g)), idx]
    start = np.where(
        idx == len(p) - 1,
        angles[np.arange(len(g)), np.maximum(cnt - 1, 0)],
        angles[np.arange(len(g)), np.minimum(idx, len(p) - 1)],
    )
    theta = start + gap / 2
    u = np.column_stack((np.cos(theta), np.sin(theta)))
    vv = violations(p, g, u).min(axis=1)
    # Include near-critical configurations as well as strictly uncovered ones.
    order = np.argsort(vv)[::-1]
    chosen = []
    seen = set()
    for k in order:
        if vv[k] < -0.012 and len(chosen) >= 100:
            break
        key = (
            *np.round(g[k] / 0.006).astype(int),
            int((theta[k] % (2 * np.pi)) / 0.04),
        )
        if key not in seen:
            chosen.append(k)
            seen.add(key)
        if len(chosen) >= 450:
            break
    chosen = np.array(chosen, dtype=int)
    witness = int(order[0])
    gg, uu = g[chosen], u[chosen]
    if refine:
        rg, ru = refine_holes(p, gg, uu, domain=domain)
        gg = np.vstack((rg, gg))
        uu = np.vstack((ru, uu))
        rv = violations(p, gg, uu).min(axis=1)
        k = int(np.argmax(rv))
        worst = float(rv[k])
        wp, wu = gg[k], uu[k]
    else:
        worst = float(vv[witness])
        wp, wu = g[witness], u[witness]
    return (
        gg,
        uu,
        dict(
            max_violation_m=worst * 1000,
            position_m=(wp * 1000).tolist(),
            normal=wu.tolist(),
            tested_positions=len(g),
            uncovered_samples=int((vv > 1e-8).sum()),
            locally_refined=refine,
        ),
    )
