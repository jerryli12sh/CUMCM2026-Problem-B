"""Apply short derivative probes to sources first found at grid search sites."""

import math
import numpy as np
from geometry import coverage_certificate
from selective_scan import SelectiveScanRobot
from gradient import VertexGradientRobot, project_polygon


class AllVertexGradientRobot(VertexGradientRobot):
    def __init__(
        self,
        *args,
        gradient_step=4.0,
        gradient_axes="dominant",
        grid_search=True,
        one_sided=False,
        grid_count=6,
        **kwargs
    ):
        super().__init__(
            *args, gradient_step=gradient_step, gradient_axes=gradient_axes, **kwargs
        )
        self.grid_search = grid_search
        self.one_sided = one_sided
        if grid_search:
            self.pending_stations = [
                (1200.0, 0.0),
                (600.0, 1050.0),
                (-600.0, 1050.0),
                (-1200.0, 0.0),
                (-600.0, -1050.0),
                (600.0, -1050.0),
            ]
            if grid_count == 8:
                self.pending_stations = [
                    (900.0, 0.0),
                    (750.0, 750.0),
                    (0.0, 900.0),
                    (-750.0, 750.0),
                    (-900.0, 0.0),
                    (-750.0, -750.0),
                    (0.0, -900.0),
                    (750.0, -750.0),
                ]
            elif grid_count == 10:
                self.pending_stations = [
                    (900.0, 0.0),
                    (900.0, 600.0),
                    (300.0, 900.0),
                    (-300.0, 900.0),
                    (-900.0, 600.0),
                    (-900.0, 0.0),
                    (-900.0, -600.0),
                    (-300.0, -900.0),
                    (300.0, -900.0),
                    (900.0, -600.0),
                ]
            assert coverage_certificate([(0.0, 0.0)] + self.pending_stations)[0]

    def scan(self, q, all_known=False, force_discovery=True):
        prior = {ch for ch, b in self.beliefs.items() if b.known}
        result = SelectiveScanRobot.scan(
            self, q, all_known=all_known, force_discovery=force_discovery
        )
        if any(abs(v / 150 - round(v / 150)) > 1e-9 for v in q):
            return result
        channels = [
            ch
            for ch, b in self.beliefs.items()
            if ch not in prior and b.known and not b.cleared and b.circle()[1] > 100
        ]
        if not channels:
            return result
        bearings = {
            ch: next(
                beta
                for p, beta in reversed(self.beliefs[ch].bearings)
                if math.dist(p, q) < 1e-7
            )
            for ch in channels
        }
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
        readings = {ch: {} for ch in channels}
        h = self.gradient_step
        for axis, sign in ((0, 1), (1, 1), (0, -1), (1, -1)):
            if self.one_sided and sign < 0:
                continue
            p = (q[0] + sign * h, q[1]) if axis == 0 else (q[0], q[1] + sign * h)
            order = [
                ch
                for ch in channels
                if axis in axes[ch] and not self.beliefs[ch].cleared
            ]
            if self.channel in order:
                order.remove(self.channel)
                order.insert(0, self.channel)
            for ch in order:
                r = self.measure(p, ch)
                if r is None:
                    r = next(
                        e["response"]
                        for e in reversed(self.log)
                        if e["path"] == "/measure"
                        and e["request"]["channel"] == ch
                        and tuple(e["request"]["position"][k] for k in ("x", "y")) == p
                    )
                if r["measure_result"] == "direction":
                    readings[ch][axis, sign] = r["svd_deg"]
        for ch, r in readings.items():
            b = self.beliefs[ch]
            if b.cleared or len(r) < (1 if self.one_sided else 2) * len(axes[ch]):
                continue
            g = np.array(
                [
                    math.radians(
                        (
                            r[a, 1]
                            - (
                                math.degrees(bearings[ch])
                                if self.one_sided
                                else r[a, -1]
                            )
                            + 180
                        )
                        % 360
                        - 180
                    )
                    / ((1 if self.one_sided else 2) * h)
                    for a in axes[ch]
                ]
            )
            gg = float(g @ g)
            if gg < 1e-12:
                continue
            if len(g) == 2:
                delta = (-g[1] / gg, g[0] / gg)
            else:
                beta = bearings[ch]
                distance = (
                    math.sin(beta) if axes[ch][0] == 0 else -math.cos(beta)
                ) / g[0]
                if distance <= 0:
                    continue
                delta = (distance * math.cos(beta), distance * math.sin(beta))
            raw = (q[0] + delta[0], q[1] + delta[1])
            if math.hypot(*raw) > 1800:
                continue
            target = project_polygon(raw, b.poly)
            self.gradient_targets[ch] = target
            self.gradient_diagnostics.append(
                dict(
                    channel=ch,
                    station=q,
                    raw_estimate=raw,
                    projected_estimate=target,
                    gradient=g.tolist(),
                )
            )
        return result
