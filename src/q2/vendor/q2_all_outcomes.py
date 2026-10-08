"""第二问修订：仅优化定位效果，允许第二次无信号。

统一处理direction / no_signal / near，未知有效半径在两次观测间保持不变。
数值示例S1=(0,0),theta1=0，完整圆扇区位于目标圆域内。
输出近优选点，不宣称位置连续域全局最优。依赖numpy。
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("previous_q2", ROOT / "solve_q2.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
clip, diameter, posterior = base.clip, base.diameter, base.posterior
ALPHA = math.pi / 180
OUTER, INNER = base.priors(24)
OUTER = clip(OUTER, -1, 0, -5 * math.cos(ALPHA))


def outside_vertices(poly, q, radius):
    """凸多边形减去圆盘后，其凸包的候选顶点：外侧顶点和边圆交点。

    圆周位于多边形内部的点不是外侧集合凸包的极点，故不需采样圆弧。
    使用闭边界；上下界调用时分别对半径作微小松弛/收紧。
    """
    if not poly:
        return []
    rr = radius * radius
    out = [p for p in poly if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= rr]
    for p, t in zip(poly, poly[1:] + poly[:1]):
        dx, dy = t[0] - p[0], t[1] - p[1]
        ux, uy = p[0] - q[0], p[1] - q[1]
        aa = dx * dx + dy * dy
        if aa == 0:
            continue
        bb, cc = 2 * (ux * dx + uy * dy), ux * ux + uy * uy - rr
        discr = bb * bb - 4 * aa * cc
        if discr < 0:
            continue
        root = math.sqrt(discr)
        for frac in ((-bb - root) / (2 * aa), (-bb + root) / (2 * aa)):
            if 0 <= frac <= 1:
                out.append((p[0] + frac * dx, p[1] + frac * dy))
    return out


def silent_bounds(q):
    c = (q[0] ** 2 + q[1] ** 2) / 2
    po = clip(OUTER, q[0], q[1], c + 1e-7)
    pi = clip(INNER, q[0], q[1], c - 1e-7)
    vo = outside_vertices(po, q, 1000 - 1e-7)
    vi = outside_vertices(pi, q, 1000 + 1e-7)
    return diameter(vi), diameter(vo)


def cap_disk(poly, q, radius, outer=True, sides=128):
    if not poly:
        return []
    offset = radius if outer else radius * math.cos(math.pi / sides)
    if all(math.hypot(p[0] - q[0], p[1] - q[1]) <= offset for p in poly):
        return poly
    if (
        max(p[0] for p in poly) < q[0] - radius
        or min(p[0] for p in poly) > q[0] + radius
    ):
        return []
    if (
        max(p[1] for p in poly) < q[1] - radius
        or min(p[1] for p in poly) > q[1] + radius
    ):
        return []
    for i in range(sides):
        angle = 2 * math.pi * i / sides
        a, b = math.cos(angle), math.sin(angle)
        poly = clip(poly, a, b, a * q[0] + b * q[1] + offset)
        if not poly:
            break
    return poly


def inside(poly, q):
    return bool(poly) and all(
        (t[0] - p[0]) * (q[1] - p[1]) - (t[1] - p[1]) * (q[0] - p[0]) >= -1e-8
        for p, t in zip(poly, poly[1:] + poly[:1])
    )


def angle_range(poly, q):
    if inside(poly, q):
        return 0.0, 2 * math.pi
    angles = sorted(math.atan2(p[1] - q[1], p[0] - q[0]) % (2 * math.pi) for p in poly)
    if not angles:
        return 0.0, 0.0
    gaps = [
        angles[(i + 1) % len(angles)] - a + (2 * math.pi if i == len(angles) - 1 else 0)
        for i, a in enumerate(angles)
    ]
    j = max(range(len(gaps)), key=gaps.__getitem__)
    start = angles[(j + 1) % len(angles)] - ALPHA
    width = min(2 * math.pi, 2 * math.pi - gaps[j] + 2 * ALPHA)
    return start, width


def direction_table(q, bins=512, lower=False, sampled_lower=False):
    po = cap_disk(OUTER, q, 1500, outer=True)
    pi = cap_disk(INNER, q, 1500, outer=False) if lower or sampled_lower else []
    start, width = angle_range(po, q)
    if width == 0:
        return dict(
            start=start,
            width=width,
            upper=np.zeros(bins),
            lower=np.zeros(bins),
            sampled_lower=0.0,
        )
    step = width / bins
    upper_values, lower_values = [], []
    sampled = 0.0
    for i in range(bins):
        beta = start + (i + 0.5) * step
        pu = posterior(po, q, beta, ALPHA + step / 2)
        upper_values.append(diameter(outside_vertices(pu, q, 5 - 1e-7)))
        if lower:
            pl = posterior(pi, q, beta, max(0, ALPHA - step / 2))
            lower_values.append(diameter(outside_vertices(pl, q, 5 + 1e-7)))
        if sampled_lower:
            pl = posterior(pi, q, beta, ALPHA)
            sampled = max(sampled, diameter(outside_vertices(pl, q, 5 + 1e-7)))
    return dict(
        start=start,
        width=width,
        upper=np.asarray(upper_values),
        lower=np.asarray(lower_values),
        sampled_lower=sampled,
    )


def worst_score(q, bins=512, incumbent=float("inf")):
    lo, hi = silent_bounds(q)
    if lo > incumbent:
        return float("inf")
    table = direction_table(q, bins)
    return max(hi, float(table["upper"].max()), 10.0)


def worst_certificate(q, bins=16384):
    silent_lo, silent_hi = silent_bounds(q)
    table = direction_table(q, bins, sampled_lower=True)
    return dict(
        point=q,
        distance_from_first_m=math.hypot(*q),
        no_signal_diameter_bounds_m=[silent_lo, silent_hi],
        direction_worst_diameter_bounds_m=[
            table["sampled_lower"],
            float(table["upper"].max()),
        ],
        all_outcomes_worst_bounds_m=[
            max(silent_lo, table["sampled_lower"]),
            max(silent_hi, float(table["upper"].max()), 10.0),
        ],
        bearing_bins=bins,
        note="near branch has diameter at most 10 m; bounds use floating arithmetic",
    )


def search_worst():
    best = (worst_score((834.0, 552.0)), (834.0, 552.0))
    checked = 0
    for a in range(-1500, 3001, 50):
        for b in range(0, 1551, 50):
            if a * a + b * b > 3000**2 or (a == 0 and b == 0):
                continue
            value = worst_score((a, b), bins=256, incumbent=best[0])
            checked += 1
            if value < best[0]:
                best = value, (float(a), float(b))
    print("Worst coarse search", best, "checked", checked, flush=True)
    step = 25.0
    for level in range(4):
        center = best[1]
        # 每层重新计算，避免将不同读数分辨率的分数混在一起比较。
        bins = 512 if level < 2 else 2048
        best = (worst_score(center, bins=bins), center)
        for i in range(-4, 5):
            for j in range(-4, 5):
                q = (center[0] + i * step, center[1] + j * step)
                value = worst_score(q, bins=bins, incumbent=best[0])
                if value < best[0]:
                    best = value, q
        step /= 4
        print("Worst refinement", level, best, flush=True)
    return best[1]


def sample_joint(n=16384, seed=20260911):
    """示例先验：源在目标圆域均匀，R在[1000,1500]均匀，误差均匀。

    条件化首次有信号与theta1=0后，(G,R)在联合可行集上按面积*dR均匀。
    以全扇区均匀位置和均匀R拒绝采样得到，不把径向距离误当成均匀。
    """
    rng = np.random.default_rng(seed)
    chunks = []
    total = 0
    while total < n:
        size = 2 * (n - total) + 64
        r = np.sqrt(rng.uniform(25, 1500**2, size))
        radius = rng.uniform(1000, 1500, size)
        phi = rng.uniform(-ALPHA, ALPHA, size)
        keep = r <= radius
        arr = np.column_stack(
            (
                r[keep] * np.cos(phi[keep]),
                r[keep] * np.sin(phi[keep]),
                radius[keep],
                rng.uniform(-ALPHA, ALPHA, int(keep.sum())),
            )
        )
        chunks.append(arr)
        total += len(arr)
    return np.concatenate(chunks)[:n]


def sample_losses(q, states, table, silent_lo, silent_hi):
    dx, dy = states[:, 0] - q[0], states[:, 1] - q[1]
    rr = np.hypot(dx, dy)
    silent = rr > states[:, 2]
    near = rr <= 5
    direction = ~silent & ~near
    lower = np.zeros(len(states))
    upper = np.zeros(len(states))
    lower[silent], upper[silent] = silent_lo, silent_hi
    upper[near] = 10
    if np.any(direction):
        beta = np.arctan2(dy[direction], dx[direction]) + states[direction, 3]
        relative = (beta - table["start"]) % (2 * math.pi)
        assert relative.max() <= table["width"] + 1e-7
        index = np.minimum(
            (relative / table["width"] * len(table["upper"])).astype(int),
            len(table["upper"]) - 1,
        )
        upper[direction] = table["upper"][index]
        if len(table["lower"]):
            lower[direction] = table["lower"][index]
    return (
        lower,
        upper,
        dict(
            no_signal=float(silent.mean()),
            near=float(near.mean()),
            direction=float(direction.mean()),
        ),
    )


def expected_score(q, states, bins=512, incumbent=float("inf")):
    if math.hypot(*q) < 1e-6:
        return diameter(OUTER)
    lo, hi = silent_bounds(q)
    prob_silent = float(
        (np.hypot(states[:, 0] - q[0], states[:, 1] - q[1]) > states[:, 2]).mean()
    )
    if prob_silent * lo > incumbent:
        return float("inf")
    table = direction_table(q, bins)
    _, loss, _ = sample_losses(q, states, table, lo, hi)
    return float(loss.mean())


def search_expected(states):
    best = (expected_score((950.0, 500.0), states), (950.0, 500.0))
    for a in range(-1500, 3001, 100):
        for b in range(0, 1501, 100):
            if a * a + b * b > 3000**2:
                continue
            value = expected_score((a, b), states, bins=512, incumbent=best[0])
            if value < best[0]:
                best = value, (float(a), float(b))
    print("Expected coarse search", best, flush=True)
    step = 25.0
    for level in range(4):
        center = best[1]
        bins = 512 if level < 2 else 2048
        best = (expected_score(center, states, bins), center)
        for i in range(-4, 5):
            for j in range(-4, 5):
                q = (center[0] + i * step, center[1] + j * step)
                value = expected_score(q, states, bins=bins, incumbent=best[0])
                if value < best[0]:
                    best = value, q
        step /= 4
        print("Expected refinement", level, best, flush=True)
    return best[1]


def expected_check(q, states, bins=16384):
    lo, hi = silent_bounds(q)
    table = direction_table(q, bins, lower=True)
    lower, upper, probabilities = sample_losses(q, states, table, lo, hi)
    return dict(
        sample_size=len(states),
        point=q,
        outcome_probabilities=probabilities,
        sample_expected_diameter_bounds_m=[float(lower.mean()), float(upper.mean())],
        lower_mean_standard_error_m=float(lower.std(ddof=1) / math.sqrt(len(states))),
        upper_mean_standard_error_m=float(upper.std(ddof=1) / math.sqrt(len(states))),
        note="Monte Carlo under an explicit illustrative prior; expectation bounds still have sampling uncertainty",
    )


def main():
    robust = search_worst()
    states = sample_joint()
    average = search_expected(states)
    holdout = sample_joint(200000, seed=20260912)
    points = {
        "previous_guaranteed_reception": (833.740234375, 552.1568813157363),
        "all_outcomes_minimax_candidate": robust,
        "illustrative_bayes_candidate": average,
        "simple_expanded_example": (950.0, 500.0),
    }
    results = {}
    for name, q in points.items():
        results[name] = dict(
            worst=worst_certificate(q),
            expected=expected_check(q, holdout),
            inside_previous_safe_region=base.safe(q),
        )
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    result = dict(
        scope="S1 at origin, theta1=0; movement excluded from objective; no-signal allowed",
        search="deterministic coarse grid plus refinement, not a global optimality certificate",
        expected_prior="uniform joint feasible density dArea(G)*dR after first measurement; second bearing error uniform",
        results=results,
    )
    (ROOT / "q2_all_outcomes_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
