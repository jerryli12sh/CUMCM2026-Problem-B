"""Teammate-inspired selective shared bearings; retain all unknown-channel scans."""

import math
from combined import CombinedRobot


class SelectiveScanRobot(CombinedRobot):
    def __init__(
        self, *args, shared_limit=3, minimum_crossing=15.0, minimum_gain=40.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.shared_limit = shared_limit
        self.minimum_crossing = minimum_crossing
        self.minimum_gain = minimum_gain
        self._shared_allowed = None
        self.shared_skips = 0

    def measure(self, q, ch):
        if (
            self._shared_allowed is not None
            and self.beliefs[ch].known
            and ch not in self._shared_allowed
        ):
            self.shared_skips += 1
            return None
        return super().measure(q, ch)

    def scan(self, q, all_known=False, force_discovery=True):
        candidates = []
        for ch, belief in self.beliefs.items():
            if not belief.known or belief.cleared:
                continue
            center, radius = belief.circle()
            distance = math.dist(center, q)
            if (
                radius <= max(self.try_radius, self.scan_known_radius)
                or distance >= 1450
            ):
                continue
            if not belief.bearings or any(
                math.dist(q, p) <= 120 for p in belief.positive
            ):
                continue
            angle = math.atan2(center[1] - q[1], center[0] - q[0])
            sine = max(abs(math.sin(angle - theta)) for _, theta in belief.bearings)
            crossing = math.degrees(math.asin(min(1.0, sine)))
            # This predicts information value only. The unchanged conservative
            # polygon, never this estimate, remains the clear certificate.
            predicted = distance * math.radians(2.01) / max(0.001, sine)
            gain = max(0.0, radius - predicted)
            if crossing >= self.minimum_crossing and gain >= self.minimum_gain:
                candidates.append((gain, ch))
        self._shared_allowed = {
            ch for _, ch in sorted(candidates, reverse=True)[: self.shared_limit]
        }
        try:
            return super().scan(q, all_known=all_known, force_discovery=force_discovery)
        finally:
            self._shared_allowed = None

    def run(self, resume=False):
        result = super().run(resume=resume)
        result["shared_measurements_skipped"] = self.shared_skips
        return result
