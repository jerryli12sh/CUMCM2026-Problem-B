"""Deterministic open-path optimization over currently observed task targets.

Only orders given points; no measurements, source truth or simulated outcomes.
"""

import math


def optimize_route(start, points, previous=()):
    n = len(points)
    if n < 2:
        length = math.dist(start, points[0]) if n else 0.0
        return list(range(n)), length, length
    pts = [start] + list(points)
    d = [[math.dist(a, b) for b in pts] for a in pts]

    def cost(o):
        return sum(d[a][b] for a, b in zip([0] + o, o))

    def nearest(first=None):
        left = set(range(1, n + 1))
        order = []
        pos = 0
        if first is not None:
            order.append(first)
            left.remove(first)
            pos = first
        while left:
            j = min(left, key=lambda j: (d[pos][j], j))
            order.append(j)
            left.remove(j)
            pos = j
        return order

    # Preserve v04's feasible candidate so the target-point proxy cannot worsen.
    original = nearest()
    base = cost(original)
    v04 = list(original)
    for _ in range(3):
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = v04[:i] + list(reversed(v04[i : j + 1])) + v04[j + 1 :]
                c = cost(candidate)
                if c < base - 1e-7:
                    v04, base = candidate, c

    def improve(o):
        o = list(o)
        for _ in range(80):
            best_delta = -1e-7
            change = None
            # Open-path two-opt, including a change of the free endpoint.
            for i in range(n - 1):
                a = o[i - 1] if i else 0
                b = o[i]
                for j in range(i + 1, n):
                    c = o[j]
                    z = o[j + 1] if j + 1 < n else None
                    delta = d[a][c] - d[a][b]
                    if z is not None:
                        delta += d[b][z] - d[c][z]
                    if delta < best_delta:
                        best_delta, change = delta, ("reverse", i, j)
            # Relocate a single vertex, which two-opt alone cannot always do.
            for i, x in enumerate(o):
                a = o[i - 1] if i else 0
                b = o[i + 1] if i + 1 < n else None
                removal = -d[a][x] + (d[a][b] - d[x][b] if b is not None else 0.0)
                rest = o[:i] + o[i + 1 :]
                for j in range(n):
                    if j == i:
                        continue
                    a = rest[j - 1] if j else 0
                    b = rest[j] if j < n - 1 else None
                    delta = (
                        removal
                        + d[a][x]
                        + (d[x][b] - d[a][b] if b is not None else 0.0)
                    )
                    if delta < best_delta:
                        best_delta, change = delta, ("move", i, j)
            if change is None:
                break
            kind, i, j = change
            if kind == "reverse":
                o[i : j + 1] = reversed(o[i : j + 1])
            else:
                o.insert(j, o.pop(i))
        return o

    seeds = [v04, original, list(reversed(original))]
    seeds.extend(
        nearest(i) for i in sorted(range(1, n + 1), key=lambda i: (d[0][i], i))[:6]
    )
    if previous:
        kept = [i + 1 for i in previous if 0 <= i < n]
        if len(set(kept)) == len(kept):
            seeds.append(kept + [i for i in original if i not in kept])
    # Cheapest insertion supplies a different family of initial paths.
    insertion = [min(range(1, n + 1), key=lambda i: (d[0][i], i))]
    remaining = set(range(1, n + 1)) - set(insertion)
    while remaining:
        choices = []
        for x in remaining:
            for j in range(len(insertion) + 1):
                a = insertion[j - 1] if j else 0
                b = insertion[j] if j < len(insertion) else None
                delta = d[a][x] + (d[x][b] - d[a][b] if b is not None else 0.0)
                choices.append((delta, x, j))
        _, x, j = min(choices)
        insertion.insert(j, x)
        remaining.remove(x)
    seeds.append(insertion)
    best = v04
    best_cost = base
    for seed in seeds:
        o = improve(seed)
        c = cost(o)
        assert len(o) == n and set(o) == set(range(1, n + 1))
        if c < best_cost - 1e-7:
            best, best_cost = o, c
    assert best_cost <= base + 1e-6
    return [i - 1 for i in best], base, best_cost
