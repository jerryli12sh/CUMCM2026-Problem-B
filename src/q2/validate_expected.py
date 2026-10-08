"""Independent checks for posterior, integration, and entire-block loss envelopes."""

import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from expectation import *
from bayes import _table, losses

ROOT = Path(__file__).resolve().parents[2]


def run():
    rng = np.random.default_rng(94017)
    checks = {}
    g = Geometry()
    n = 75000
    for model in MODELS:
        x, rho, e, z = samples(g, n, 94018, model)
        r = np.linalg.norm(x, axis=1)
        integ = lambda t: t * (1 if t <= 1000 or model == "flat" else (1500 - t) / 500)
        expected = quad(integ, 5, 1000)[0] / quad(integ, 5, 1500, points=[1000])[0]
        observed = float(np.mean(r <= 1000))
        assert abs(observed - expected) < 0.008
        a = np.maximum(1000, r)
        u = (rho - a) / (1500 - a)
        assert abs(u.mean() - 0.5) < 0.008 and np.all(rho >= r)
        checks[model + "_posterior"] = {
            "n": n,
            "p_le1000_analytic": expected,
            "empirical": observed,
            "rho_conditional_uniform_mean": float(u.mean()),
        }
    # All feedbacks include their generating source, including range/error endpoints.
    x, rho, e, z = samples(g, n, 94019)
    q = rng.uniform([-1200, -1400], [2700, 1400], (n, 2))
    e2 = rng.uniform(-A, A, n)
    d = np.linalg.norm(x - q, axis=1)
    k = np.where(d > rho, 0, np.where(d <= 5, 1, 2))
    b = np.arctan2(x[:, 1] - q[:, 1], x[:, 0] - q[:, 0]) + e2
    valid = np.zeros(n, bool)
    for j in range(n):
        valid[j] = g.contains(x[j], q[j], int(k[j]), float(b[j]))
    assert valid.all()
    checks["truth_containment"] = {"n": n, "failures": int((~valid).sum())}
    # Source-conditional analytical average vs direct rho/error sampling.
    xx, rr, ee, zz = samples(g, 131072, 94020)
    q = [870, 530]
    t = table(g, q, bins=65536)
    l, h = envelopes(g, q, t, xx)
    a, b, k = losses(
        g, q, _table(g, q, 8192), xx, rr, error_dist("uniform").ppf(norm.cdf(zz))
    )
    diff = float(((a + b) - (l + h)).mean() / 2)
    se = float(np.std((a + b - l - h) / 2) / math.sqrt(len(xx)))
    assert abs(diff) < 5 * se + 0.03
    checks["rao_blackwell_vs_full_MC"] = {
        "difference": diff,
        "paired_se": se,
        "n": len(xx),
    }
    # Independent per-source endpoint feedback integral must stay in block envelope.
    # Direct quadrature over rho and e at sampled Q; no use of angular_mean.
    rows = []
    for S, theta, q, delta in [
        ((0, 0), 0, [870, 530], 8),
        ((0, 0), 0, [250, 2], 15),
        ((1501, 0), 0, [220, 190], 4),
    ]:
        gg = Geometry(S, theta)
        xs = samples(gg, 16, 94021)[0]
        tt = table(gg, q, delta, bins=8192)
        ll, hh = envelopes(gg, q, tt, xs)
        for shift in [[0, 0], [delta, 0], [-delta, 0], [0, delta], [0, -delta]]:
            qq = np.array(q) + shift
            pt = _table(gg, qq, 16384)
            for i, source in enumerate(xs):
                es = np.linspace(-A, A, 257)
                aa = max(1000, np.linalg.norm(source))
                rrs = np.linspace(aa, 1500, 257)
                # Independent 2-D product integration; endpoints retained.
                xx2 = np.tile(source, (len(es) * len(rrs), 1))
                r2 = np.repeat(rrs, len(es))
                e22 = np.tile(es, len(rrs))
                dl, dh, kk = losses(gg, qq, pt, xx2, r2, e22)
                ml = float(dl.mean())
                mh = float(dh.mean())
                # Finite product-rule approximation has O(1/257) bounded variation error.
                quadrature_slack = 2 * gg.D / 256
                assert ml >= ll[i] - quadrature_slack and mh <= hh[i] + quadrature_slack
            # Stronger direct point interval check at identical source nodes.
            tight = table(gg, qq, bins=65536)
            pl, ph = envelopes(gg, qq, tight, xs)
            assert np.all(pl >= ll - 1e-6) and np.all(ph <= hh + 1e-6)
        rows.append(
            {"S": S, "q": q, "delta": delta, "sources": 16, "station_positions": 5}
        )
    checks["block_envelopes"] = rows
    for q in [[0, 0], [4000, 2000]]:
        t = table(g, q, bins=2048)
        lo, hi = envelopes(g, q, t, xx[:128])
        assert np.all(lo >= g.Dlower - 1e-6) and np.all(hi <= g.D + 1e-6)
    checks["same_location_and_no_information"] = True
    # Worst-feedback witness extraction for Q4 selected point.
    old = json.loads((ROOT / "data/q2/reference/q4_standard_regions.json").read_text())[
        "best"
    ]["q"]
    tt = _table(g, old, 65536)
    j = int(np.argmax(tt["lower"]))
    beta = tt["start"] + (j + 0.5) * tt["width"] / len(tt["lower"])
    witness = []
    for label, poly in [
        ("normal", g.branch(old, 2, beta, upper=False)),
        ("silent", g.branch(old, 0, upper=False)),
    ]:
        ds = squareform(pdist(poly))
        a, b = np.unravel_index(ds.argmax(), ds.shape)
        p = np.array([poly[a], poly[b]])
        witness.append(
            {
                "feedback": label,
                "beta_deg": float(np.degrees(beta)) if label == "normal" else None,
                "diameter_m": float(ds[a, b]),
                "sources_local": p.tolist(),
                "first_distances_m": np.linalg.norm(p, axis=1).tolist(),
            }
        )
    checks["q4_worst_witnesses"] = witness
    checks["passed"] = True
    (ROOT / "outputs/q2/validation_expected.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2)
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
