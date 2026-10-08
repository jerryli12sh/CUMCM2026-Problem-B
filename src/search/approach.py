"""Take an earlier transverse bearing before committing to an uncertain range."""

import math
from routing import RoutingRobot


class ApproachRobot(RoutingRobot):
    def __init__(self, *args, approach_fraction=0.5, transverse_offset=90.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.approach_fraction = approach_fraction
        self.transverse_offset = transverse_offset

    def finish_source(self, ch):
        b = self.beliefs[ch]
        center, radius = b.circle()
        if not b.cleared and len(b.bearings) == 1 and radius > 250:
            d = math.dist(self.position, center)
            if d > max(35, radius * 0.6):
                theta = b.bearings[0][1]
                base = tuple(
                    self.position[i]
                    + self.approach_fraction * (center[i] - self.position[i])
                    for i in range(2)
                )
                q = (
                    base[0] - self.transverse_offset * math.sin(theta),
                    base[1] + self.transverse_offset * math.cos(theta),
                )
                self.measure(q, ch)
        return super().finish_source(ch)
