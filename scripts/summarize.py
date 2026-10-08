"""Recompute headline metrics from the retained CSV rows."""

import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    formal = list(csv.DictReader((ROOT / "data/formal_results.csv").open()))
    out = {"formal": {}}
    for question in ["3", "4"]:
        rr = [r for r in formal if r["question"] == question]
        n = np.array([int(r["cleared_sources"]) for r in rr])
        x = np.array([float(r["seconds_per_source"]) for r in rr])
        out["formal"][question] = dict(
            cases=len(rr),
            cleared_sources=int(n.sum()),
            equal_case_mean=float(x.mean()),
            source_weighted_mean=float(n @ x / n.sum()),
        )
    rr = list(csv.DictReader((ROOT / "data/q3/paired_300.csv").open()))
    groups = {
        m: sorted([r for r in rr if r["method"] == m], key=lambda r: int(r["index"]))
        for m in ["baseline", "short_baseline"]
    }
    assert [r["index"] for r in groups["baseline"]] == [
        r["index"] for r in groups["short_baseline"]
    ]
    x, y = [np.array([float(r["average_time_s"]) for r in groups[m]]) for m in groups]
    delta = x - y
    half = t.ppf(0.975, len(delta) - 1) * delta.std(ddof=1) / len(delta) ** 0.5
    assert all(
        r["all_cleared"] == "True" and r["source_count"] == r["cleared_count"]
        for r in rr
    )
    out["q3_paired"] = dict(
        cases=len(delta),
        sources_per_method=sum(int(r["source_count"]) for r in groups["baseline"]),
        baseline_mean=float(x.mean()),
        method_mean=float(y.mean()),
        saving_s=float(delta.mean()),
        saving_percent=float((x.mean() - y.mean()) / x.mean() * 100),
        saving_ci95=[float(delta.mean() - half), float(delta.mean() + half)],
        faster_cases=int((delta > 0).sum()),
    )
    cores = json.loads((ROOT / "data/q2/standard_taper_cores.json").read_text())
    row = next(r for r in cores["rows"] if r["epsilon"] == 0.05)
    regions = json.loads((ROOT / "data/q2/standard_taper_regions.json").read_text())
    out["q2"] = dict(
        recommended_point_m=regions["best"]["q"],
        expected_diameter_interval_m=[regions["best"]["lo"], regions["best"]["hi"]],
        threshold_m=row["threshold_m"],
        candidate_area_m2=row["area_inner_m2"],
        safe_radius_lower_m=row["max_radius_lower_m"],
        centers_m=[r["center_local_m"] for r in row["components"]],
    )
    qr = list(csv.DictReader((ROOT / "data/q4/episodes.csv").open()))
    ref = json.loads((ROOT / "data/q4/stratified_reference.json").read_text())
    out["q4_local"] = {}
    rng = np.random.default_rng(2026091344)
    for group, size in [("I", 10), ("J", 30)]:
        arms = ["v5", "simplified"] + (["combined"] if group == "I" else [])
        groups = {
            a: sorted(
                [r for r in qr if r["group"] == group and r["arm"] == a],
                key=lambda r: r["case"],
            )
            for a in arms
        }
        base = groups["v5"]
        draws = []
        for d in range(1, 14):
            ids = np.array([i for i, r in enumerate(base) if int(r["D"]) == d])
            assert len(ids) == size
            draws.append(ids[rng.integers(0, size, (40000, size))])
        for arm, values in groups.items():
            assert [r["case"] for r in values] == [r["case"] for r in base]
            assert all(int(r["N"]) == int(r["cleared"]) == 13 for r in values)
            values = np.array([float(r["seconds_per_source"]) for r in values])
            ci = np.quantile(
                sum(values[ii].mean(axis=1) for ii in draws) / 13, [0.025, 0.975]
            )
            assert abs(values.mean() - ref["statistics"][group][arm]["mean"]) < 1e-9
            assert np.max(np.abs(ci - ref["statistics"][group][arm]["mean95"])) < 1e-9
            out["q4_local"][group + "/" + arm] = dict(
                cases=len(values), mean=float(values.mean()), ci95=ci.tolist()
            )
    for r in qr:
        components = sum(
            float(r[k])
            for k in ["travel", "measure", "switch", "clear_success", "clear_failure"]
        )
        assert abs(components - float(r["seconds_per_source"])) < 1e-5
    return out


if __name__ == "__main__":
    result = summarize()
    dest = ROOT / "results/summary.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
