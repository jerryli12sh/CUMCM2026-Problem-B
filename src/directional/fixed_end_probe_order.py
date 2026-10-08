"""Reorder paid six-metre probes while preserving the final point and channel.

Only ordinary bearings are used. Each source receives the same probe as before.
The original ordering is always a candidate; service fees are constant, while
travel and the initial channel switch are charged exactly in microseconds.
"""

import math


def choose(start, current_channel, channels, points):
    channels = list(channels)
    if len(channels) < 3:
        return channels, 0.0
    if len(channels) != len(set(channels)):
        raise ValueError("Duplicate probe channel")

    def edge(a, b):
        return round(math.dist(a, b) / 5 * 1e6)

    def cost(order):
        return (
            edge(start, points[order[0]])
            + 1000000 * (current_channel != order[0])
            + sum(
                edge(points[a], points[b]) + 1000000 for a, b in zip(order, order[1:])
            )
        )

    baseline = cost(channels)
    candidates = [channels]
    last = channels[-1]
    free = channels[:-1]
    n = len(free)
    if n <= 10:
        # Held--Karp: the final source remains fixed, so the subsequent policy
        # sees the original position and active channel.
        dp = {}
        for j, ch in enumerate(free):
            dp[(1 << j, j)] = (
                edge(start, points[ch]) + 1000000 * (current_channel != ch),
                (ch,),
            )
        for mask in range(1, 1 << n):
            for j in range(n):
                if (mask, j) not in dp:
                    continue
                value, path = dp[(mask, j)]
                for k, ch in enumerate(free):
                    if mask & (1 << k):
                        continue
                    key = (mask | (1 << k), k)
                    option = (
                        value + edge(points[free[j]], points[ch]) + 1000000,
                        path + (ch,),
                    )
                    if key not in dp or option < dp[key]:
                        dp[key] = option
        candidates += [list(dp[((1 << n) - 1, j)][1]) + [last] for j in range(n)]
    else:
        # Bounded work for unusually many simultaneous first discoveries.
        ordered = sorted(
            free,
            key=lambda ch: (
                math.atan2(points[ch][1] - start[1], points[ch][0] - start[0]),
                ch,
            ),
        )
        for direction in (ordered, ordered[::-1]):
            for i in range(n):
                candidates.append(direction[i:] + direction[:i] + [last])
    best = min(candidates, key=cost)
    saving = baseline - cost(best)
    return (best, saving / 1e6) if saving > 0 else (channels, 0.0)


def ordered_items(bot, q, prior):
    items = list(bot.beliefs.items())
    if not getattr(bot, "fixed_end_probe_order", False):
        return items
    if not all(abs(x / 150 - round(x / 150)) < 1e-9 for x in q):
        return items
    eligible = []
    points = {}
    for ch, b in items:
        if ch in prior or not b.known or b.cleared or b.circle()[1] <= 100:
            continue
        beta = next(
            beta for point, beta in reversed(b.bearings) if math.dist(point, q) < 1e-6
        )
        point = (q[0] - 6 * math.sin(beta), q[1] + 6 * math.cos(beta))
        if (ch, float(point[0]), float(point[1])) in bot.measured:
            return items
        eligible.append(ch)
        points[ch] = point
    order, saving = choose(bot.position, bot.channel, eligible, points)
    if order == eligible:
        return items
    bot.probe_order_diagnostics.append(
        dict(
            station=q,
            start=bot.position,
            current_channel=bot.channel,
            original=eligible,
            chosen=order,
            points=points,
            saving_s=saving,
            fixed_final=eligible[-1],
        )
    )
    # Noneligible sources would be skipped by the original loop.
    return [(ch, bot.beliefs[ch]) for ch in order] + [
        (ch, b) for ch, b in items if ch not in eligible
    ]
