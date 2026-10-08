"""第二问的保守选点与数值验证，Python标准库。

局部坐标：第一检测点为原点，示向度为+x，距离单位米。
不假定源距离或误差分布。示例忽略目标圆域截断，适用于S1在圆心的情况；
对其他S1，候选点的安全保证仍成立，但利用圆域可进一步改善选点。
搜索是确定性网格加局部加密，不宣称空间全局最优。
最终用角度区间覆盖给出已选点的最坏直径上下界。
"""

import json
import math
from pathlib import Path

ALPHA = math.pi / 180
RMIN, RMAX = 1000.0, 1500.0


def clip(poly, a, b, c):
    if not poly:
        return []
    result = []
    p = poly[-1]
    fp = a * p[0] + b * p[1] - c
    for q in poly:
        fq = a * q[0] + b * q[1] - c
        if (fp <= 0) != (fq <= 0):
            t = fp / (fp - fq)
            result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
        if fq <= 0:
            result.append(q)
        p, fp = q, fq
    return result


def diameter(poly):
    best = 0.0
    for i, p in enumerate(poly):
        for q in poly[:i]:
            best = max(best, (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
    return math.sqrt(best)


def priors(arc_steps=16):
    """外切多边形包含真实扇区；内接多边形为正常初测可行位置的子集。"""
    h = RMAX * math.tan(ALPHA)
    outer = [(0.0, 0.0), (RMAX, -h), (RMAX, h)]
    angles = [-ALPHA + 2 * ALPHA * i / arc_steps for i in range(arc_steps + 1)]
    for angle in angles:
        outer = clip(outer, math.cos(angle), math.sin(angle), RMAX)
    inner = [(0.0, 0.0)] + [(RMAX * math.cos(t), RMAX * math.sin(t)) for t in angles]
    # x>5保证距离S1>5；边界的极小正裕量仅用于避开near状态。
    inner = clip(inner, -1, 0, -(5 + 1e-7))
    return outer, inner


def posterior(prior, q, bearing, alpha=ALPHA):
    lo, hi = bearing - alpha, bearing + alpha
    a, b = math.sin(lo), -math.cos(lo)
    p = clip(prior, a, b, a * q[0] + b * q[1])
    a, b = -math.sin(hi), math.cos(hi)
    return clip(p, a, b, a * q[0] + b * q[1])


def safe(q, tol=1e-8):
    x, y = q
    centers = [
        (0, 0),
        (RMIN * math.cos(ALPHA), RMIN * math.sin(ALPHA)),
        (RMIN * math.cos(ALPHA), -RMIN * math.sin(ALPHA)),
    ]
    return all(math.hypot(x - a, y - b) <= RMIN + tol for a, b in centers)


def max_lateral(a, budget):
    if a < 0 or a > min(budget, RMIN):
        return -1.0
    disc = RMIN**2 - (a - RMIN * math.cos(ALPHA)) ** 2
    if disc < 0:
        return -1.0
    return min(
        math.sqrt(max(0, min(budget, RMIN) ** 2 - a**2)),
        math.sqrt(disc) - RMIN * math.sin(ALPHA),
    )


def angular_range(prior, q):
    # 本数值示例搜索位于完整先验上方的点，故所有方向在(-pi,0)，没有跨周问题。
    assert q[1] > max(p[1] for p in prior)
    angles = [math.atan2(p[1] - q[1], p[0] - q[0]) for p in prior]
    return min(angles) - ALPHA, max(angles) + ALPHA


def sampled_score(prior, q, bins=128):
    low, high = angular_range(prior, q)
    step = (high - low) / bins
    return max(
        diameter(posterior(prior, q, low + (i + 0.5) * step)) for i in range(bins)
    )


def certify(q, outer, inner, bins=8192):
    """任何读数位于一个区间内；将半角扩大区间半宽给出全区间的保守上界。

    内接先验在精确采样读数下给出真实最坏直径的下界；外切先验与扩角
    给出上界。因此此处不是把有限采样最大值冒充最坏情况保证。
    该界仍受普通双精度舍入误差影响，并非区间算术机器认证。
    """
    low, high = angular_range(outer, q)
    step = (high - low) / bins
    lower = upper = 0.0
    for i in range(bins):
        bearing = low + (i + 0.5) * step
        lower = max(lower, diameter(posterior(inner, q, bearing)))
        upper = max(upper, diameter(posterior(outer, q, bearing, ALPHA + step / 2)))
    return dict(
        lower_bound_m=lower,
        upper_bound_m=upper,
        gap_m=upper - lower,
        bearing_bins=bins,
        bin_width_deg=math.degrees(step),
    )


def search(budget, outer):
    best = (float("inf"), None, None, None)

    def evaluate(a, fraction, bins=128):
        nonlocal best
        if not 0 < fraction <= 1:
            return
        b = max_lateral(a, budget) * fraction
        if b <= RMAX * math.tan(ALPHA) + 5 or not safe((a, b)):
            return
        score = sampled_score(outer, (a, b), bins)
        if score < best[0]:
            best = (score, (a, b), a, fraction)

    a_max = min(budget, RMIN)
    for i in range(1, 32):
        for j in range(1, 17):
            evaluate(a_max * i / 32, j / 16)
    if best[1] is None:
        raise ValueError("本示例的候选子区域内没有点；应放宽几何筛选或增加预算")
    da, df = a_max / 32, 1 / 16
    for _ in range(4):
        ca, cf = best[2], best[3]
        for i in range(-4, 5):
            for j in range(-4, 5):
                evaluate(ca + i * da / 4, cf + j * df / 4)
        da /= 4
        df /= 4
    # 更细的读数采样重新检查局部候选，防止粗采样改变排序。
    ca, cf = best[2], best[3]
    best = (float("inf"), None, None, None)
    for i in range(-4, 5):
        for j in range(-2, 3):
            evaluate(ca + i * a_max / 1024, cf + j / 1024, bins=1024)
    return best[1]


def measured_crossing_lower(q):
    a, b = q[0], abs(q[1])
    h = RMAX * math.tan(ALPHA)
    ratio = min(b / a if a > 0 else float("inf"), (b - h) / (RMAX - a))
    return max(0.0, math.degrees(math.atan(ratio)) - 1)


def main():
    outer, inner = priors()
    rows = []
    for budget in [200, 400, 600, 800, 1000]:
        q = search(budget, outer)
        assert safe(q) and math.hypot(*q) <= budget + 1e-7
        row = dict(
            movement_budget_m=budget,
            second_point_local_m=q,
            move_distance_m=math.hypot(*q),
            move_and_measure_time_s=math.hypot(*q) / 5 + 5,
            guaranteed_measured_crossing_deg=measured_crossing_lower(q),
            worst_diameter=certify(q, outer, inner),
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    example = (750.0, 600.0)
    example_result = dict(
        second_point_local_m=example,
        guaranteed_reception=safe(example),
        guaranteed_measured_crossing_deg=measured_crossing_lower(example),
        worst_diameter=certify(example, outer, inner),
    )
    result = dict(
        scope="S1=(0,0), theta1=0; conservative uncertainty, no distribution assumptions",
        optimization="grid and local refinement; no claim of globally optimal station position",
        rows=rows,
        simple_example=example_result,
    )
    path = Path(__file__).with_name("q2_results.json")
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Saved", path, flush=True)


if __name__ == "__main__":
    main()
