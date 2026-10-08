"""Independent expected-loss search and frozen-partition validation."""

import argparse, csv, json, math, time, heapq
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from expectation import Geometry, samples, table, envelopes, confidence, point
from expected_cores import cores
from source_witness import lower as source_lower
from moments import stats as fast_stats
from geometry_view import NormalGeometryView

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/q2"
EPS = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
STAGES = [32768, 131072, 524288, 2097152]
_FORK_EVALUATE = None


def _fork_evaluate(z):
    return _FORK_EVALUATE(z)


def save(path, d):
    path.write_text(
        json.dumps(
            d,
            ensure_ascii=False,
            indent=2,
            default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist(),
        )
    )


def setup(args):
    states = {"standard": ((0, 0), 0), "boundary": ((1501, 0), 0)}
    S, theta = states[args.scenario]
    if args.first:
        S, theta = args.first, args.bearing
    g = Geometry(S, theta)
    name = args.scenario + "_" + args.model
    if args.first:
        name = "custom_" + args.model
    return g, name


def search(g, name, model):
    x = samples(g, 4096, 61014, model)[0]
    coarse = []
    t0 = time.time()

    def fun(q, bins=4096):
        t = table(g, q, bins=bins)
        lo, hi = envelopes(g, q, t, x)
        return float((lo.mean() + hi.mean()) / 2)

    a, b, c, d = g.bounds
    sym = bool(np.linalg.norm(g.S) < 1e-12)
    if sym:
        b = 0
    for i, qx in enumerate(np.linspace(a, c, 25)):
        for qy in np.linspace(b, d, 19):
            q = [float(qx), float(qy)]
            coarse.append({"q": q, "mean": fun(q)})
    seeds = sorted(coarse, key=lambda z: z["mean"])[:6]
    # Reuse old point as an extra seed, never as a search constraint.
    old = (
        json.loads(
            (
                ROOT
                / "data/q2/reference"
                / ("q4_" + name.split("_")[0] + "_regions.json")
            ).read_text()
        )
        if name.split("_")[0] in ["standard", "boundary"]
        else None
    )
    if old:
        seeds.append({"q": old["best"]["q"]})
    local = []
    for s in seeds:
        opt = minimize(
            lambda q: fun(q, 8192) if a <= q[0] <= c and b <= q[1] <= d else g.D,
            s["q"],
            method="Nelder-Mead",
            options={"maxiter": 130, "xatol": 0.15, "fatol": 0.003},
        )
        local.append({"q": opt.x.tolist(), "mean": float(opt.fun)})
    best = min(local, key=lambda z: z["mean"])
    opt = minimize(
        lambda q: fun(q, 32768),
        best["q"],
        method="Nelder-Mead",
        options={"maxiter": 90, "xatol": 0.05, "fatol": 0.001},
    )
    best = {"q": opt.x.tolist(), "mean": float(opt.fun)}
    out = {
        "name": name,
        "model": model,
        "scenario": {"S": g.S.tolist(), "theta": g.theta},
        "best_development": best,
        "coarse": coarse,
        "local": local,
        "source_sample_n": 4096,
        "seed": 61014,
        "seconds": time.time() - t0,
    }
    save(DATA / (name + "_search.json"), out)
    print(name, "search", best, "seconds", round(time.time() - t0), flush=True)
    return out


def freeze(g, name, model, budget=16000, minimum=2, resume_partition=False):
    info = json.loads((DATA / (name + "_search.json")).read_text())
    x = samples(g, 4096, 61014, model)[0]
    # Independent pilot validates ONLY the already-locked selected point, not partition selection.
    xp = samples(g, STAGES[-1], 72015, model)[0]
    stage_rows = []
    for n in STAGES:
        r = point(g, info["best_development"]["q"], xp[:n], bins=131072, family=16)
        stage_rows.append(r)
        if r["hi"] - r["lo"] <= 0.005 * ((r["hi"] + r["lo"]) / 2):
            break
    U = r["hi"]
    thresholds = U * (1 + EPS)
    if thresholds[-1] >= g.Dlower:
        raise ValueError("Threshold includes unbounded no-information region")
    info.update(
        best=r,
        point_stages=stage_rows,
        thresholds=thresholds.tolist(),
        eps=EPS.tolist(),
    )
    save(DATA / (name + "_search.json"), info)
    a, b, c, d = g.bounds
    sym = bool(np.linalg.norm(g.S) < 1e-12)
    if sym:
        b = 0
    queue = []
    serial = 0
    leaves = []
    count = 0
    t0 = time.time()

    def push(z):
        nonlocal serial
        serial += 1
        # Priority changes processing order only; all unvisited blocks remain in partition.
        # Focus resolution on neighborhoods identified by the full-domain exploratory pass.
        bx, by = info["best_development"]["q"]
        scale = 60 if g.D > 500 else 12
        targets = [(bx, by)] + ([(bx, -by)] if not sym else [])
        dist2 = min(
            max(z[0] - tx, 0, tx - z[2]) ** 2 + max(z[1] - ty, 0, ty - z[3]) ** 2
            for tx, ty in targets
        )
        priority = (z[2] - z[0]) * (z[3] - z[1]) / (1 + (dist2 / scale**2) ** 3)
        heapq.heappush(queue, (-priority, serial, z))

    previous_blocks = 0
    if resume_partition:
        previous = json.loads((DATA / (name + "_partition.json")).read_text())
        previous_blocks = previous["developed_blocks"]
        for z in previous["cells"]:
            push(z)
    else:
        push([a, b, c, d])
    while queue and count < budget:
        _, _, z = heapq.heappop(queue)
        q = [(z[0] + z[2]) / 2, (z[1] + z[3]) / 2]
        delta = math.hypot(z[2] - z[0], z[3] - z[1]) / 2
        # Development-only block bounds; ALL terminal blocks independently validated later.
        t = table(g, q, delta, bins=32768 if resume_partition and delta < 3 else 8192)
        lo, hi = envelopes(g, q, t, x)
        l = float(lo.mean())
        u = float(hi.mean())
        count += 1
        se = 4 * max(float(np.std(lo)), float(np.std(hi))) / math.sqrt(len(x)) + 0.03
        crossing = np.any((thresholds >= l - se) & (thresholds <= u + se))
        strict = np.any((thresholds[:2] >= l - se) & (thresholds[:2] <= u + se))
        target = (0.25 if g.D < 500 else 0.5) if strict else minimum
        if crossing and max(z[2] - z[0], z[3] - z[1]) > target:
            ax, ay, cx, cy = z
            mx, my = q
            for zz in [
                [ax, ay, mx, my],
                [mx, ay, cx, my],
                [ax, my, mx, cy],
                [mx, my, cx, cy],
            ]:
                push(zz)
        else:
            leaves.append(z)
        if count % 2000 == 0:
            print(
                name,
                "freeze",
                count,
                "queue",
                len(queue),
                "sec",
                round(time.time() - t0),
                flush=True,
            )
    leaves += [z for _, _, z in queue]
    save(
        DATA / (name + "_partition.json"),
        {
            "cells": leaves,
            "developed_blocks": count + previous_blocks,
            "budget_limited_blocks": len(queue),
            "mirror": sym,
            "thresholds": thresholds.tolist(),
            "selection_seed": 61014,
            "point_seed": 72015,
            "seconds": time.time() - t0,
        },
    )
    print(
        name,
        "frozen leaves",
        len(leaves),
        "pending development",
        len(queue),
        flush=True,
    )


def validate_regions(g, name, model):
    info = json.loads((DATA / (name + "_search.json")).read_text())
    part = json.loads((DATA / (name + "_partition.json")).read_text())
    cells = part["cells"]
    # Recompute pilot under explicitly split alpha=.025, before independent region validation.
    xp = samples(g, STAGES[-1], 72015, model)[0]
    stage_rows = []
    for n in STAGES:
        anchor = point(g, info["best_development"]["q"], xp[:n], bins=131072, family=16)
        stage_rows.append(anchor)
        if anchor["hi"] - anchor["lo"] <= 0.005 * (anchor["hi"] + anchor["lo"]) / 2:
            break
    info.update(
        best=anchor,
        point_stages=stage_rows,
        thresholds=(anchor["hi"] * (1 + EPS)).tolist(),
    )
    save(DATA / (name + "_search.json"), info)
    thresholds = np.array(info["thresholds"])
    del xp
    # Bonferroni covers both envelopes, every fixed block, all four sample stages and four main runs.
    family = 2 * 4 * 4 * len(cells)
    use_coarse = g.D > 500
    if use_coarse:
        family *= 2
    x = samples(g, STAGES[-1], 83016, model)[0]
    if use_coarse:
        g = NormalGeometryView(g, arc=12)
    outpath = DATA / (name + "_validated_half.csv")
    done = []
    if outpath.exists():
        raise FileExistsError("Use a fresh outputs/q2 folder for region validation")
    if outpath.exists():
        done = (
            np.genfromtxt(outpath, delimiter=",", skip_header=1)
            .reshape(-1, 10)
            .tolist()
        )
    f = outpath.open("a" if done else "w")
    w = csv.writer(f)
    if not done:
        w.writerow(
            [
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "lower_m",
                "upper_m",
                "n",
                "bins",
                "geometry_gap_m",
                "sampling_width_m",
            ]
        )
    t0 = time.time()

    def evaluate(z):
        q = [(z[0] + z[2]) / 2, (z[1] + z[3]) / 2]
        delta = math.hypot(z[2] - z[0], z[3] - z[1]) / 2
        bins = 32768 if delta < 4 else 8192
        t = table(g, q, delta, bins=bins)

        def measure(t, n):
            if g.D > 500 and delta >= 2:
                lo, hi = envelopes(g, q, t, x[:n])
                lo = np.maximum(lo, source_lower(g, q, delta, x[:n], t["silent"][0]))
                assert np.all(lo <= hi + 1e-6)
                return confidence(lo, hi, t["bound"], family, alpha=0.025)
            return fast_stats(g, q, t, x[:n], family, alpha=0.025)

        for j, n in enumerate(STAGES):
            r = measure(t, n)
            crossing = np.any((thresholds >= r["lo"]) & (thresholds <= r["hi"]))
            if not crossing:
                break
            if bins < 32768 and delta < 4:
                bins = 32768
                t = table(g, q, delta, bins=bins)
                r = measure(t, n)
            geom = r["geom_hi"] - r["geom_lo"]
            err = r["error_lo"] + r["error_hi"]
            uncertain = (thresholds >= r["lo"]) & (thresholds <= r["hi"])
            geometric = (thresholds >= r["geom_lo"]) & (thresholds <= r["geom_hi"])
            if np.all(~uncertain | geometric):
                break
            # If geometry already spans a threshold and sampling is small relative to geometry,
            # keep the block pending rather than spend samples pretending to fix spatial error.
            if geom > 0 and err <= max(0.06, 0.15 * geom):
                break
        row = [
            *z,
            r["lo"],
            r["hi"],
            n,
            bins,
            r["geom_hi"] - r["geom_lo"],
            r["error_lo"] + r["error_hi"],
        ]
        return row

    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import os

    workers = int(os.environ.get("Q41_WORKERS", "2"))
    start_index = len(done)
    evaluate_fn = evaluate
    if os.environ.get("Q41_EXECUTOR") == "process":
        # Optional Unix acceleration: fork after read-only source arrays and native
        # kernels are initialized. Each child runs exactly the same block function.
        import multiprocessing

        global _FORK_EVALUATE
        _FORK_EVALUATE = evaluate
        pool_factory = lambda: ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("fork")
        )
        evaluate_fn = _fork_evaluate
    else:
        pool_factory = lambda: ThreadPoolExecutor(max_workers=workers)
    with pool_factory() as pool:
        for start in range(start_index, len(cells), 128):
            for row in pool.map(evaluate_fn, cells[start : start + 128]):
                w.writerow(row)
                done.append(row)
            f.flush()
            if start // 512 != (start + 128) // 512 or len(done) == len(cells):
                print(
                    name,
                    "validated",
                    len(done),
                    "/",
                    len(cells),
                    "sec",
                    round(time.time() - t0),
                    flush=True,
                )
    f.close()
    arr = np.array(done)[:, :6]
    if part["mirror"]:
        mirror = arr.copy()
        mirror[:, 1] = -arr[:, 3]
        mirror[:, 3] = -arr[:, 1]
        arr = np.vstack([arr, mirror])
    np.savetxt(
        DATA / (name + "_cells.csv"),
        arr,
        delimiter=",",
        header="xmin,ymin,xmax,ymax,lower_m,upper_m",
        comments="",
    )
    areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    L = float(arr[:, 4].min())
    rows = []
    for eps, d in zip(EPS, thresholds):
        good = arr[:, 5] <= d
        possible = arr[:, 4] <= d
        rows.append(
            {
                "epsilon": float(eps),
                "threshold_m": float(d),
                "area_inner_m2": float(areas[good].sum()),
                "area_outer_m2": float(areas[possible].sum()),
                "pending_area_m2": float(areas[possible & ~good].sum()),
                "global_ratio_bound": float(d / L) if L > 0 else None,
            }
        )
    result = {
        "name": name,
        "model": model,
        "objective": "expected_hard_feasible_diameter",
        "scenario": info["scenario"],
        "best": info["best"],
        "global_lower_m": L,
        "rows": rows,
        "F1_bounds": [g.Dlower, g.D],
        "unprocessed_blocks": 0,
        "budget_limited_development_blocks": part["budget_limited_blocks"],
        "full_domain_partition": True,
        "confidence_family": family,
        "validation_seed": 83016,
        "validation_status": "simultaneous model-conditional statistical bounds plus float64 geometric enclosures; not interval arithmetic certificate",
        "seconds": time.time() - t0,
    }
    save(DATA / (name + "_regions.json"), result)
    cores(name, 0.5)
    print(
        name,
        "DONE",
        [
            (r["epsilon"], round(r["area_inner_m2"]), round(r["area_outer_m2"]))
            for r in rows
        ],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["standard", "boundary"], default="standard")
    p.add_argument("--model", choices=["taper", "flat"], default="taper")
    p.add_argument(
        "--phase", choices=["search", "freeze", "validate", "all"], default="all"
    )
    p.add_argument("--budget", type=int, default=16000)
    p.add_argument("--resume-partition", action="store_true")
    p.add_argument("--first", type=float, nargs=2)
    p.add_argument("--bearing", type=float, default=0)
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    g, name = setup(a)
    if a.phase in ["search", "all"]:
        search(g, name, a.model)
    if a.phase in ["freeze", "all"]:
        freeze(g, name, a.model, a.budget, resume_partition=a.resume_partition)
    if a.phase in ["validate", "all"]:
        validate_regions(g, name, a.model)


if __name__ == "__main__":
    main()
