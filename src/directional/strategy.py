"""C20 combined policy for directional discovery, posterior clearance, and rolling routes."""

import json, math
from base_strategy import C20Robot, make_robot as make_base_robot, HERE
from route_optimizer import optimize_route
from certify_layout_integer import certify
from robot import Robot
from gradient import project_polygon
from forward_anchor_recovery import GridProbeMixin
from uniform_field_posterior import infer


class DirectionalSearchRobot(C20Robot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recover_other_corners = True
        self.grid_probe_diagnostics = []
        self.uniform_cache = {}
        self.preclear = "utility"
        self.preclear_diagnostics = []
        self.particle_cache = None
        self.regional_diagnostics = []
        self.scan_dp = False
        self.recovery_mode = "retreat2"
        self.clear_utility = 0.0
        self.deferred_channels = {}
        self.shift = 0.0
        self.catalog = json.loads((HERE / "catalog_lattice20.json").read_text())
        self.regional_anchor = True
        self.catalog_cost_diagnostics = []
        self.convex_slide = "all"
        self.convex_slide_diagnostics = []
        self.next_scan = None
        self.route_anchor = True
        self.route_anchor_diagnostics = []
        self.nonzero_prior_diagnostics = []
        self.fixed_end_probe_order = True
        self.probe_order_diagnostics = []
        self.clear_utility_diagnostics = []

    def scan(self, q, *args, **kwargs):
        initial = not self.discovery_stations
        before = set(self.gradient_targets)
        result = GridProbeMixin.scan(self, q, *args, **kwargs)
        for ch in set(self.gradient_targets) - before:
            target, diagnostic, posterior = infer(
                self.beliefs[ch], self.gradient_targets[ch]
            )
            if posterior is not None:
                self.uniform_cache[ch] = (self.beliefs[ch].version, posterior)
            self.gradient_targets[ch] = target
            self.posterior_diagnostics.append(
                dict(
                    channel=ch,
                    **{
                        k: v
                        for k, v in diagnostic.items()
                        if k
                        in (
                            "used",
                            "reason",
                            "baseline_success_mass",
                            "candidate_success_mass",
                            "displacement_m",
                            "chosen",
                        )
                    }
                )
            )
        if initial:
            known = [
                self.source_target(c)
                for c, b in self.beliefs.items()
                if b.known and not b.cleared
            ]
            best = None
            for item in self.catalog:
                points = [tuple(p) for p in item["points"]]
                if len(points) != 20 or points[0] != (0.0, 0.0):
                    raise ValueError("Invalid catalog layout")
                order, _, length = optimize_route(self.position, known + points[1:])
                from catalog_anchor_cost import cost

                fee = cost(self, known, points, order)
                self.catalog_cost_diagnostics.append(
                    dict(degrees=item["degrees"], path_m=length, expected_anchor_s=fee)
                )
                length += 5 * fee
                if best is None or length < best[0]:
                    best = (length, points, item)
            if best is None:
                raise ValueError("Empty catalog")
            proof = certify(
                [[round(v * 1000) for v in p] for p in best[1]], domain_mm=1770000
            )
            if not proof["certified"]:
                raise ValueError("Selected layout is not certified")
            self.required_stations = best[1]
            self.pending_stations = best[1][1:].copy()
            self.geometry_certificate = proof
            self.catalog_selected = best[2]["degrees"]
            self.route_keys = []
        return result

    def choose_joint_task(self):
        task = super().choose_joint_task()
        if task:
            from regional_route import select

            task = select(self, task)
            from convex_station_slide import select as slide

            task = slide(self, task)
        return task

    def finish_source(self, ch):
        if ch in self.gradient_targets and not self.beliefs[ch].cleared:
            from uniform_field_posterior import clearance

            clearance(self, ch, 50.0)
        if (
            self.preclear
            and ch in self.gradient_targets
            and not self.beliefs[ch].cleared
        ):
            from approach_clear_trials import attempt

            taken, cleared, original = attempt(self, ch)
            if taken:
                if cleared:
                    del self.gradient_targets[ch]
                    self.gradient_guesses += 1
                    if self.preclear_diagnostics[-1]["results"][0]:
                        self.gradient_first_clear_successes += 1
                    return
                return self.retreat_finish(
                    ch, target_override=original, had_earlier_failure=True
                )
        if (
            self.clear_utility
            and ch in self.gradient_targets
            and not self.beliefs[ch].cleared
        ):
            from uniform_field_posterior import clearance

            clearance(self, ch, self.clear_utility)
        if (
            self.recovery_mode
            and ch in self.gradient_targets
            and not self.beliefs[ch].cleared
        ):
            return self.retreat_finish(ch)
        if self.shift and ch in self.gradient_targets:
            q = self.source_target(ch)
            toward = self.position
            if self.shift < 0 and len(self.route_keys) > 1:
                key = self.route_keys[1]
                end = self.source_target(key[1]) if key[0] == "source" else key[1]
                v = tuple(end[i] - self.position[i] for i in (0, 1))
                den = sum(x * x for x in v)
                t = (
                    max(
                        0,
                        min(
                            1,
                            sum((q[i] - self.position[i]) * v[i] for i in (0, 1)) / den,
                        ),
                    )
                    if den
                    else 0
                )
                toward = tuple(self.position[i] + t * v[i] for i in (0, 1))
            d = math.dist(toward, q)
            shift = min(abs(self.shift), d)
            if d > 1e-9:
                self.gradient_targets[ch] = project_polygon(
                    tuple(q[i] + shift * (toward[i] - q[i]) / d for i in (0, 1)),
                    self.beliefs[ch].poly,
                )
        return super().finish_source(ch)

    def retreat_finish(self, ch, target_override=None, had_earlier_failure=False):
        b = self.beliefs[ch]
        anchor = b.positive[-1]
        target = self.source_target(ch) if target_override is None else target_override
        del self.gradient_targets[ch]
        self.gradient_guesses += 1
        if self.clear(target, ch, safe=False):
            if not had_earlier_failure:
                self.gradient_first_clear_successes += 1
            return
        if self.recovery_mode.startswith("bayes"):
            from correlated_bearing_posterior import infer

            for _ in range(2 if self.recovery_mode == "bayes2" else 1):
                point, diag = infer(b, self.position, minimum_gain=0.03)
                mass = diag.get("candidate_success_mass", 0.0)
                distance = math.dist(point, self.position)
                take = bool(
                    diag["used"]
                    and mass >= 0.4
                    and 2 < distance < 70
                    and (distance / 5 + 3) < 30 * mass
                )
                self.bayes_clear_diagnostics.append(
                    dict(
                        channel=ch, point=point, mass=mass, distance=distance, take=take
                    )
                )
                if not take:
                    break
                if self.clear(point, ch, safe=False):
                    return
            self.measure(self.position, ch)
            if b.cleared:
                return
            if self.march:
                self.march_source(ch)
            if not b.cleared:
                return self._conservative_finish(ch)
            return
        response = Robot.measure(self, self.position, ch)
        if b.cleared:
            return
        if response and response["measure_result"] == "no_signal":
            # A directional source can disappear after the approach overshoots
            # it. Cheap optical trials are independent of emission direction.
            length = math.dist(anchor, target)
            if length > 100:
                steps = (32.0, 64.0) if self.recovery_mode == "retreat2" else (32.0,)
                for distance in steps:
                    q = tuple(
                        target[i] + distance * (anchor[i] - target[i]) / length
                        for i in (0, 1)
                    )
                    self.retreat_trials += 1
                    if self.clear(q, ch, safe=False):
                        self.retreat_successes += 1
                        return
                Robot.measure(self, self.position, ch)
                if b.cleared:
                    return
        if self.march:
            self.march_source(ch)
        if not b.cleared:
            return self._conservative_finish(ch)

    def run(self, resume=False):
        result = super().run(resume)
        result.update(
            strategy="C20_combined",
            grid_probe_diagnostics=self.grid_probe_diagnostics,
            regional_diagnostics=self.regional_diagnostics,
            preclear_diagnostics=self.preclear_diagnostics,
            catalog_cost_diagnostics=self.catalog_cost_diagnostics,
            probe_order_diagnostics=self.probe_order_diagnostics,
            clear_utility_diagnostics=self.clear_utility_diagnostics,
            convex_slide_diagnostics=self.convex_slide_diagnostics,
            route_anchor_diagnostics=self.route_anchor_diagnostics,
            nonzero_prior_diagnostics=self.nonzero_prior_diagnostics,
        )
        return result


def make_robot(transport, robot_id="private-research", audit=None):
    config = json.loads((HERE / "settings.json").read_text())
    settings = dict(config["route_settings"], layout_choice=False)
    return DirectionalSearchRobot(
        transport,
        robot_id=robot_id,
        audit=audit,
        mode="joint_smart",
        ring_radius=1130,
        try_radius=150,
        discovery_gain=150,
        **settings
    )
