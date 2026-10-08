"""Certified 20-station initialization and shared directional completion logic."""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "search"))
from policy import OfficialQ4Robot
from route_optimizer import optimize_route
from certify_layout_integer import certify
from correlated_bearing_posterior import infer
from robot import Robot


class C20Robot(OfficialQ4Robot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.catalog = json.loads((HERE / "catalog20.json").read_text())
        initial = json.loads((HERE / "layout20_short.json").read_text())
        self.required_stations = [tuple(p) for p in initial["points"]]
        assert (
            self.required_stations[0] == (0.0, 0.0)
            and len(self.required_stations) == 20
        )
        self.geometry_certificate = certify(
            [[round(v * 1000) for v in p] for p in self.required_stations],
            domain_mm=1770000,
        )
        if not self.geometry_certificate["certified"]:
            raise ValueError("Initial layout is not certified")
        self.pending_stations = self.required_stations[1:].copy()
        self.catalog_selected = None
        self.posterior_diagnostics = []
        self.retreat_trials = 0
        self.retreat_successes = 0

    def scan(self, q, *args, **kwargs):
        initial = not self.discovery_stations
        before = set(self.gradient_targets)
        result = super().scan(q, *args, **kwargs)
        for ch in set(self.gradient_targets) - before:
            target, diagnostic = infer(
                self.beliefs[ch],
                self.gradient_targets[ch],
                minimum_gain=0.03,
                use_mean=True,
            )
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
                _, _, length = optimize_route(self.position, known + points[1:])
                if best is None or length < best[0]:
                    best = (length, points, item)
            if best is None:
                raise ValueError("Empty catalog")
            points = best[1]
            proof = certify(
                [[round(v * 1000) for v in p] for p in points], domain_mm=1770000
            )
            if not proof["certified"]:
                raise ValueError("Selected layout is not certified")
            self.required_stations = points
            self.pending_stations = points[1:].copy()
            self.geometry_certificate = proof
            self.catalog_selected = best[2]["degrees"]
            self.route_keys = []
        return result

    def finish_source(self, ch):
        if ch not in self.gradient_targets or self.beliefs[ch].cleared:
            return super().finish_source(ch)
        b = self.beliefs[ch]
        anchor = b.positive[-1]
        target = self.source_target(ch)
        del self.gradient_targets[ch]
        self.gradient_guesses += 1
        if self.clear(target, ch, safe=False):
            self.gradient_first_clear_successes += 1
            return
        response = Robot.measure(self, self.position, ch)
        if b.cleared:
            return
        if response and response["measure_result"] == "no_signal":
            length = math.dist(anchor, target)
            if length > 100:
                for distance in (32.0, 64.0):
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
            coverage_kind="C20_1770m_integer_certificate_actual_receipts_or_16_known",
            required_stations=self.required_stations,
            actual_discovery_stations=self.discovery_stations,
            catalog_selected=self.catalog_selected,
            posterior_diagnostics=self.posterior_diagnostics,
            retreat_trials=self.retreat_trials,
            retreat_successes=self.retreat_successes,
        )
        return result


def make_robot(transport, robot_id="private-research", audit=None):
    config = json.loads((HERE / "settings.json").read_text())
    settings = dict(config["route_settings"], layout_choice=False)
    return C20Robot(
        transport,
        robot_id=robot_id,
        audit=audit,
        mode="joint_smart",
        ring_radius=1130,
        try_radius=150,
        discovery_gain=150,
        **settings
    )
