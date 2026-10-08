"""Numerical equivalence of optional accelerators and expectation-witness check."""

import json, math, time
from pathlib import Path
import numpy as np
from expectation import *
from fast import value
from batch import values
from moments import stats
from source_witness import lower, _lower as numpy_witness

ROOT = Path(__file__).resolve().parents[2]


def run():
    rng = np.random.default_rng(719)
    maxerr = 0
    for i in range(80):
        g = Geometry((0, 0) if i % 2 else (1501, 0))
        q = rng.uniform([-500, -600], [1800, 900])
        p = fast_prepare(disk(g.outer, q, 1500, True, 512))
        a = float(rng.uniform(0.001, 0.3))
        rad = float(rng.uniform(0, 30))
        bins = 1024
        vv = values(p, q, a, rad, bins)
        old = np.array(
            [
                value(p, q, -math.pi + (j + 0.5) * 2 * math.pi / bins, a, rad)
                for j in range(bins)
            ]
        )
        err = float(np.max(abs(vv - old)))
        assert err < 1e-7
        maxerr = max(maxerr, err)
    g = Geometry()
    x = samples(g, 524288, 15727)[0]
    moment_error = 0
    for q, delta in [
        ([907, 577], 0.5),
        ([0, 0], 0),
        ([0, 0], 10),
        ([600, 2], 8),
        ([2500, 1800], 30),
    ]:
        t = table(g, q, delta, 8192)
        l, h = envelopes(g, q, t, x)
        a = confidence(l, h, t["bound"], 3000, 0.025)
        b = stats(g, q, t, x, 3000)
        err = max(abs(a[k] - b[k]) for k in a)
        assert err < 1e-8
        moment_error = max(moment_error, err)
    x = samples(g, 24, 1555)[0]
    rng = np.random.default_rng(1554)
    count = 0
    for i in range(40):
        q = rng.uniform([-900, -1200], [2600, 1200])
        delta = float(rng.choice([2, 20, 50, 150]))
        t = table(g, q, delta, 4096)
        l = lower(g, q, delta, x, t["silent"][0])
        for j in range(3):
            a = rng.uniform(0, 2 * np.pi)
            qq = q + delta * np.array([np.cos(a), np.sin(a)])
            pt = table(g, qq, bins=65536)
            pl, ph = envelopes(g, qq, pt, x)
            assert np.all(l <= ph + 1e-7)
            count += len(x)
    witness_error = 0
    for first, q, delta in [
        ((0, 0), [1019, 583], 3),
        ((0, 0), [907, 577], 40),
        ((0, 0), [2500, 100], 150),
        ((1501, 0), [220, -150], 5),
        ((1200, 400), [500, -600], 45),
    ]:
        gg = Geometry(first)
        xx = samples(gg, 131072, 23817)[0]
        tt = table(gg, q, delta, 8192)
        pure = np.concatenate(
            [
                numpy_witness(gg, q, delta, xx[i : i + 8192], tt["silent"][0])
                for i in range(0, len(xx), 8192)
            ]
        )
        native = lower(gg, q, delta, xx, tt["silent"][0])
        err = float(np.max(abs(pure - native)))
        assert err < 1e-7
        witness_error = max(witness_error, err)
    out = {
        "passed": True,
        "batch_cases": 80,
        "batch_max_difference": maxerr,
        "moment_cases": 5,
        "moment_n_each": 524288,
        "moments_max_difference": moment_error,
        "source_witness_cases": count,
        "native_witness_cases": 5,
        "native_witness_n_each": 131072,
        "native_witness_max_difference": witness_error,
    }
    (ROOT / "outputs/q2/validation_acceleration.json").write_text(
        json.dumps(out, indent=2)
    )
    print(out)


if __name__ == "__main__":
    run()
