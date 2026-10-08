"""Select a visible grid anchor and perform a perpendicular short-baseline probe."""

import math
from robot import Robot
from gradient import project_polygon


def on_grid(q):
    return all(abs(x / 150 - round(x / 150)) < 1e-9 for x in q)


class GridProbeMixin:
    recover_other_corners = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_probe_diagnostics = []

    def scan(self, q, all_known=False, force_discovery=True):
        if not force_discovery:
            return
        prior = {c for c, b in self.beliefs.items() if b.known}
        channels = [c for c, b in self.beliefs.items() if not b.known and not b.cleared]
        if self.channel in channels:
            channels.remove(self.channel)
            channels.insert(0, self.channel)
        for c in channels:
            self.measure(q, c)
        if q not in self.discovery_stations:
            self.discovery_stations.append(tuple(q))
        from nonzero_population import source_type_priors

        source_type_priors(self)
        primary_anchor = None
        if getattr(self, "route_anchor", False):
            from route_anchor_choice import choose

            primary_anchor = choose(self, q, prior)
        from fixed_end_probe_order import ordered_items

        for c, b in ordered_items(self, q, prior):
            if c in prior or not b.known or b.cleared or b.circle()[1] <= 100:
                continue
            anchor = tuple(q)
            beta = next(
                beta
                for point, beta in reversed(b.bearings)
                if math.dist(point, q) < 1e-6
            )
            nearest = (float(150 * round(q[0] / 150)), float(150 * round(q[1] / 150)))
            if primary_anchor is not None:
                nearest = primary_anchor
            attempts = []
            last_result = None
            if math.dist(nearest, q) > 1e-6:
                nodes = [nearest]
                if self.recover_other_corners:
                    x = 150 * math.floor(q[0] / 150)
                    y = 150 * math.floor(q[1] / 150)
                    others = [
                        (float(x + dx), float(y + dy))
                        for dx in (0, 150)
                        for dy in (0, 150)
                    ]
                    aim = (q[0] + 1100 * math.cos(beta), q[1] + 1100 * math.sin(beta))
                    nodes += sorted(
                        (p for p in others if p != nearest),
                        key=lambda p: (math.dist(p, aim), math.dist(p, q), p),
                    )
                for node in nodes:
                    start = self.position
                    response = Robot.measure(self, node, c)
                    last_result = (
                        response["measure_result"] if response else "already_measured"
                    )
                    attempts.append(
                        dict(
                            node=node,
                            start=start,
                            response=last_result,
                            action_prefix=len(self.log),
                        )
                    )
                    if b.cleared:
                        break
                    if (
                        response is not None
                        and response["measure_result"] == "direction"
                    ):
                        anchor = node
                        beta = math.radians(response["svd_deg"])
                        break
                    # A previously paid direction at this exact node is just
                    # as usable; no synthetic or inferred reading is created.
                    existing = [
                        angle
                        for point, angle in b.bearings
                        if math.dist(point, node) < 1e-9
                    ]
                    if existing:
                        anchor = node
                        beta = existing[-1]
                        break
                self.probe_anchor_diagnostics.append(
                    dict(
                        channel=c,
                        station=q,
                        anchor=anchor,
                        result=last_result,
                        grid_origin_available=on_grid(anchor),
                        attempts=attempts,
                    )
                )
            if b.cleared:
                continue
            if not on_grid(anchor):
                self.grid_probe_diagnostics.append(
                    dict(
                        channel=c,
                        station=q,
                        attempts=attempts,
                        action="skip_optional_probe_no_visible_grid",
                        anchor=None,
                    )
                )
                continue
            sign = getattr(self, "_current_probe_signs", {}).get(c, 1)
            point = (
                anchor[0] - sign * 6 * math.sin(beta),
                anchor[1] + sign * 6 * math.cos(beta),
            )
            response = self.measure(point, c)
            self.grid_probe_diagnostics.append(
                dict(
                    channel=c,
                    station=q,
                    attempts=attempts,
                    action="probe_from_grid",
                    anchor=anchor,
                    probe=point,
                    response=(
                        response["measure_result"] if response else "already_measured"
                    ),
                )
            )
            if (
                response is None
                or response["measure_result"] != "direction"
                or b.cleared
            ):
                continue
            delta = math.radians(
                (response["svd_deg"] - math.degrees(beta) + 180) % 360 - 180
            )
            if sign * delta >= -1e-12:
                continue
            distance = -sign * 6 / delta
            raw = (
                anchor[0] + distance * math.cos(beta),
                anchor[1] + distance * math.sin(beta),
            )
            if math.hypot(*raw) <= 1800:
                self.gradient_targets[c] = project_polygon(raw, b.poly)
                self.gradient_ranges[c] = distance
                self.gradient_diagnostics.append(
                    dict(channel=c, station=q, anchor=anchor, raw_estimate=raw)
                )

    def run(self, resume=False):
        result = super().run(resume)
        result["grid_probe_diagnostics"] = self.grid_probe_diagnostics
        return result
