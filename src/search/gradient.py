"""Use the recovered grid's zero vertex gradient for speculative initial fixes.

All estimates only rank routes or attempt clears. They never replace conservative
beliefs or certify absence/clear. Failure retains the original finite fallback.
"""

import math
import numpy as np
from selective_scan import SelectiveScanRobot


def project_polygon(point, poly):
    p = np.asarray(point, float)
    v = np.asarray(poly, float)
    inside = all(
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -1e-8
        for a, b in zip(v, np.roll(v, -1, axis=0))
    )
    if inside:
        return tuple(p)
    best = None
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        e = b - a
        t = np.clip((p - a) @ e / max(e @ e, 1e-18), 0, 1)
        q = a + t * e
        if best is None or np.linalg.norm(q - p) < best[0]:
            best = (np.linalg.norm(q - p), tuple(q))
    return best[1]


class VertexGradientRobot(SelectiveScanRobot):
    def __init__(
        self,
        *args,
        gradient_step=10.0,
        extrapolate=False,
        gradient_axes="both",
        **kwargs
    ):
        kwargs.setdefault("shared_limit", 20)
        super().__init__(*args, **kwargs)
        self.gradient_step = gradient_step
        self.extrapolate = extrapolate
        self.gradient_axes = gradient_axes
        self.gradient_targets = {}
        self.gradient_guesses = 0
        self.gradient_first_clear_successes = 0
        self.gradient_diagnostics = []
        self.gradient_done = False

    def scan(self, q, all_known=False, force_discovery=True):
        initial = self.counter == 1 and not self.discovery_stations
        result = super().scan(q, all_known=all_known, force_discovery=force_discovery)
        if not initial or self.gradient_done:
            return result
        self.gradient_done = True
        channels = [
            ch
            for ch, b in self.beliefs.items()
            if b.known and not b.cleared and len(b.bearings) == 1
        ]
        readings = {ch: {} for ch in channels}
        bearings = {ch: self.beliefs[ch].bearings[0][1] for ch in channels}
        axes = {
            ch: (
                (0, 1)
                if self.gradient_axes == "both"
                else (
                    (0,)
                    if abs(math.sin(bearings[ch])) >= abs(math.cos(bearings[ch]))
                    else (1,)
                )
            )
            for ch in channels
        }
        steps = [self.gradient_step] + (
            [2 * self.gradient_step] if self.extrapolate else []
        )
        # Four short common legs; every still-active source uses public bearings.
        for h in steps:
            for axis, sign in ((0, 1), (1, 1), (0, -1), (1, -1)):
                p = (sign * h, 0.0) if axis == 0 else (0.0, sign * h)
                order = list(channels)
                if self.channel in order:
                    order.remove(self.channel)
                    order.insert(0, self.channel)
                for ch in order:
                    if self.beliefs[ch].cleared or axis not in axes[ch]:
                        continue
                    r = self.measure(p, ch)
                    if r and r["measure_result"] == "direction":
                        readings[ch][h, axis, sign] = r["svd_deg"]
        for ch, r in readings.items():
            b = self.beliefs[ch]
            if b.cleared or len(r) < 2 * len(axes[ch]) * len(steps):
                continue
            gradients = []
            for h in steps:
                gradients.append(
                    np.array(
                        [
                            math.radians((r[h, a, 1] - r[h, a, -1] + 180) % 360 - 180)
                            / (2 * h)
                            for a in axes[ch]
                        ]
                    )
                )
            g = gradients[0] if len(gradients) == 1 else 2 * gradients[0] - gradients[1]
            gg = float(g @ g)
            if gg < 1e-12:
                continue
            if len(g) == 2:
                raw = (-g[1] / gg, g[0] / gg)
            else:
                beta = bearings[ch]
                distance = (
                    math.sin(beta) if axes[ch][0] == 0 else -math.cos(beta)
                ) / g[0]
                if distance <= 0:
                    continue
                raw = (distance * math.cos(beta), distance * math.sin(beta))
            if math.hypot(*raw) > 1800:
                continue
            target = project_polygon(raw, b.poly)
            self.gradient_targets[ch] = target
            self.gradient_diagnostics.append(
                dict(
                    channel=ch,
                    raw_estimate=raw,
                    projected_estimate=target,
                    gradient=g.tolist(),
                )
            )
        return result

    def source_target(self, ch):
        if ch in self.gradient_targets and not self.beliefs[ch].cleared:
            return project_polygon(self.gradient_targets[ch], self.beliefs[ch].poly)
        return super().source_target(ch)

    def finish_source(self, ch):
        if ch in self.gradient_targets and not self.beliefs[ch].cleared:
            target = self.source_target(ch)
            del self.gradient_targets[ch]
            self.gradient_guesses += 1
            if self.clear(target, ch, safe=False):
                self.gradient_first_clear_successes += 1
                return
            self.measure(self.position, ch)
        return super().finish_source(ch)

    def run(self, resume=False):
        result = super().run(resume=resume)
        result.update(
            gradient_guesses=self.gradient_guesses,
            gradient_first_clear_successes=self.gradient_first_clear_successes,
            gradient_diagnostics=self.gradient_diagnostics,
        )
        return result
