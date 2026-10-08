"""Repeatable endpoint, relative-angle block and batch-kernel cross-checks."""

import sys, json, math
from pathlib import Path
import numpy as np
from expectation import *
from fast import value
from batch import values

ROOT = Path(__file__).resolve().parents[2]


def run():
    g = Geometry()
    cases = 0
    for r1 in [5.000001, 999.999999, 1000, 1000.000001, 1499.999999, 1500]:
        for e1 in [-A, 0, A]:
            source = r1 * np.array([np.cos(e1), np.sin(e1)])
            for rho in [max(1000, r1), 1500]:
                for d2 in [4.999999, 5, 5.000001, rho - 1e-6, rho, rho + 1e-6]:
                    q = source - np.array([0, d2])
                    for e2 in [-A, A]:
                        k, b = g.feedback(
                            source[None, :], np.array([rho]), q, np.array([e2])
                        )
                        assert g.contains(source, q, int(k[0]), float(b[0]))
                        cases += 1
    x = samples(g, 20, 12344)[0]
    rng = np.random.default_rng(12345)
    checked = 0
    for i in range(48):
        q = rng.uniform([-500, -950], [2100, 950])
        delta = float(rng.choice([0.5, 5, 20, 60, 150]))
        t = table(g, q, delta, 4096)
        l, h = envelopes(g, q, t, x)
        for j in range(3):
            phi = rng.uniform(0, 2 * np.pi)
            qq = q + delta * np.array([np.cos(phi), np.sin(phi)])
            tt = table(g, qq, bins=32768)
            ll, hh = envelopes(g, qq, tt, x)
            assert np.all(ll >= l - 1e-5) and np.all(hh <= h + 1e-5)
            checked += len(x)
    p = fast_prepare(g.outer)
    q = [900, 500]
    a = values(p, q, A, 5, 8192)
    b = np.array(
        [
            value(p, q, -math.pi + (j + 0.5) * 2 * math.pi / 8192, A, 5)
            for j in range(8192)
        ]
    )
    assert np.allclose(a, b, atol=1e-10, rtol=1e-12)
    out = {
        "passed": True,
        "endpoint_cases": cases,
        "block_source_station_cases": checked,
        "batch_max_difference": float(np.max(abs(a - b))),
    }
    (ROOT / "outputs/q2/validation_extra.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    run()
