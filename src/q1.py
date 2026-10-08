"""B题第一问：方向扇区交集、直径与等直径圆覆盖判定。

仅用 Python 标准库。输入 JSON: {"observations": [[x,y,theta_deg], ...]}。
运行：python3 solve_q1.py --self-test
      python3 solve_q1.py input.json
默认误差半宽为题设的 1 度；不擅自添加距离或目标圆域截断。
浮点实现使用米制绝对容差；近退化情形须结合 warnings 提高精度复核。
"""

import argparse
import itertools
import json
import math
import random
from pathlib import Path


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def constraints(observations, error_deg=1.0):
    """返回单位法向量形式 a*x+b*y<=c。扇区顶点取闭包。"""
    if not observations or not 0 < error_deg < 90:
        raise ValueError("需要至少一条观测，并满足 0 < error_deg < 90")
    rows = []
    for observation in observations:
        if len(observation) != 3 or not all(math.isfinite(v) for v in observation):
            raise ValueError("观测应为三个有限数 [x, y, theta_deg]")
        x, y, theta = observation
        lo, hi = [math.radians((theta + d) % 360) for d in (-error_deg, error_deg)]
        for a, b in ((math.sin(lo), -math.cos(lo)), (-math.sin(hi), math.cos(hi))):
            rows.append((a, b, a * x + b * y))
    return rows


def feasible(p, rows, tol):
    return all(a * p[0] + b * p[1] <= c + tol for a, b, c in rows)


def diameter(vertices):
    """直接遍历顶点对；小规模时便于检查，时间复杂度 O(m^2)。"""
    best, pair = 0.0, (vertices[0], vertices[0])
    for a, b in itertools.combinations(vertices, 2):
        d = dist(a, b)
        if d > best:
            best, pair = d, (a, b)
    return best, pair


def solve(observations, error_deg=1.0, tol=1e-8):
    rows = constraints(observations, error_deg)
    vertices, warnings = [], set()
    for (a, b, c), (d, e, f) in itertools.combinations(rows, 2):
        det = a * e - b * d
        if det == 0.0:
            continue
        # 不以固定的小行列式阈值直接删除近乎平行但确实相交的边界。
        if abs(det) < 1e-10:
            warnings.add("存在近乎平行的边界；近退化结果建议用高精度复核")
        p = ((c * e - b * f) / det, (a * f - c * d) / det)
        if all(math.isfinite(v) for v in p) and feasible(p, rows, tol):
            if not any(dist(p, q) <= tol for q in vertices):
                vertices.append(p)

    # 本接口的交集包含于一个开角小于180度的扇区，故不含直线。
    # 非空且不含直线的多面集必有顶点；因此枚举不到可行顶点意味着空集。
    if not vertices:
        return dict(
            status="empty", diameter=None, vertices=[], warnings=sorted(warnings)
        )

    # 检查衰退锥 A*d<=0；非零衰退方向可在某约束的边界方向上找到。
    for a, b, _ in rows:
        for direction in ((b, -a), (-b, a)):
            if all(c * direction[0] + d * direction[1] <= 1e-12 for c, d, _ in rows):
                return dict(
                    status="unbounded",
                    diameter="Infinity",
                    recession_direction=direction,
                    vertices=vertices,
                    warnings=sorted(warnings),
                )

    center = tuple(sum(p[k] for p in vertices) / len(vertices) for k in (0, 1))
    vertices.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    d, (a, b) = diameter(vertices)
    midpoint = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    radius_required_at_midpoint = max(dist(p, midpoint) for p in vertices)
    return dict(
        status="bounded",
        vertices=vertices,
        diameter=d,
        diameter_endpoints=(a, b),
        diameter_circle_center=midpoint,
        diameter_circle_radius=d / 2,
        diameter_circle_covers=radius_required_at_midpoint <= d / 2 + tol,
        radius_required_at_midpoint=radius_required_at_midpoint,
        warnings=sorted(warnings),
    )


def minimum_enclosing_circle(vertices, tol=1e-8):
    """教学用精确候选枚举，浮点计算；O(m^4)，用于小规模验证。

    最小圆由一个点、两个点的直径圆或三个不共线点的外接圆决定。
    """
    if not vertices:
        raise ValueError("空集没有本接口定义的最小包围圆")
    candidates = [(p, 0.0) for p in vertices]
    for a, b in itertools.combinations(vertices, 2):
        c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        candidates.append((c, dist(a, b) / 2))
    for a, b, c in itertools.combinations(vertices, 3):
        u, v = (b[0] - a[0], b[1] - a[1]), (c[0] - a[0], c[1] - a[1])
        det = 2 * cross(u, v)
        if det == 0:
            continue
        uu, vv = u[0] ** 2 + u[1] ** 2, v[0] ** 2 + v[1] ** 2
        center = (
            a[0] + (uu * v[1] - vv * u[1]) / det,
            a[1] + (u[0] * vv - v[0] * uu) / det,
        )
        candidates.append((center, dist(center, a)))
    valid = [
        (c, r) for c, r in candidates if all(dist(c, p) <= r + tol for p in vertices)
    ]
    c, r = min(valid, key=lambda t: t[1])
    return dict(center=c, radius=r)


def equilateral_example():
    s = math.sqrt(3)
    return [[-1218, -6 * s, 1], [618, -606 * s, 121], [600, 612 * s, 241]]


def clip_reference(rows, bound):
    """独立核验：用已知足够大的方框作初始多边形逐半平面裁剪。"""
    polygon = [(-bound, -bound), (bound, -bound), (bound, bound), (-bound, bound)]
    for a, b, c in rows:
        out = []
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            fp, fq = a * p[0] + b * p[1] - c, a * q[0] + b * q[1] - c
            pin, qin = fp <= 0, fq <= 0
            if pin != qin:
                t = fp / (fp - fq)
                out.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
            if qin:
                out.append(q)
        polygon = out
    return polygon


def self_test():
    checks = 0
    example = solve(equilateral_example())
    assert example["status"] == "bounded" and len(example["vertices"]) == 3
    assert math.isclose(example["diameter"], 36, abs_tol=1e-8)
    assert not example["diameter_circle_covers"]
    mec = minimum_enclosing_circle(example["vertices"])
    assert math.isclose(mec["radius"], 12 * math.sqrt(3), abs_tol=1e-8)
    assert dist(mec["center"], (0, 0)) < 1e-8
    checks += 1
    cases = [
        ([[0, 0, 0]], "unbounded"),
        ([[0, 0, 0], [0, 1, 0]], "unbounded"),
        ([[0, 0, 0], [0, 1, 180]], "empty"),
        ([[0, 0, 0], [0, 0, 180]], "bounded"),
        ([[0, 0, 1], [100, 0, 181]], "bounded"),
    ]
    for observations, status in cases:
        assert solve(observations)["status"] == status
        checks += 1
    assert solve(cases[3][0])["diameter"] < 1e-8
    assert math.isclose(solve(cases[4][0])["diameter"], 100, abs_tol=1e-8)
    square = solve([[-500, 0, 1], [10, -500, 91], [510, 10, 181], [0, 510, 271]])
    assert math.isclose(square["diameter"], 10 * math.sqrt(2), abs_tol=1e-8)
    assert square["diameter_circle_covers"]
    checks += 1
    rng = random.Random(20260910)
    bounded_count = 0
    for _ in range(200):
        source = (rng.uniform(-200, 200), rng.uniform(-200, 200))
        obs = []
        for _ in range(rng.randint(2, 9)):
            angle, radius = rng.uniform(0, 2 * math.pi), rng.uniform(100, 1200)
            x, y = source[0] + radius * math.cos(angle), source[1] + radius * math.sin(
                angle
            )
            theta = math.degrees(
                math.atan2(source[1] - y, source[0] - x)
            ) + rng.uniform(-1, 1)
            obs.append([x, y, theta % 360])
        result = solve(obs)
        assert result["status"] != "empty"
        if result["status"] == "bounded":
            bound = 100 + 2 * max(abs(x) for p in result["vertices"] for x in p)
            reference = clip_reference(constraints(obs), bound)
            assert math.isclose(
                diameter(reference)[0], result["diameter"], rel_tol=1e-7, abs_tol=1e-6
            )
            invariant = solve(obs + [obs[0]])
            assert math.isclose(invariant["diameter"], result["diameter"], abs_tol=1e-7)
            wrapped = solve([[x, y, t + 360] for x, y, t in obs])
            assert math.isclose(wrapped["diameter"], result["diameter"], abs_tol=1e-7)
            circle = minimum_enclosing_circle(result["vertices"])
            assert (
                result["diameter"] / 2 - 1e-7
                <= circle["radius"]
                <= result["diameter"] / math.sqrt(3) + 1e-7
            )
            bounded_count += 1
        checks += 1
    return dict(
        passed=True,
        cases=checks,
        random_cases=200,
        independently_checked_bounded_cases=bounded_count,
        example=example,
        example_minimum_enclosing_circle=mec,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        output = self_test()
    elif args.input:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        output = solve(data["observations"], data.get("error_deg", 1.0))
    else:
        output = solve(equilateral_example())
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
