"""v10: replan after a long localization approach reveals a new bearing."""

import math
from routing import RoutingRobot


class ReplanAfterBearing(Exception):
    pass


class EventRobot(RoutingRobot):
    def __init__(self, *args, replan_distance=100.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.replan_distance = replan_distance
        self._finishing = False
        self.replans = 0
        self._localization_steps = {}

    def measure(self, q, ch):
        distance = math.dist(self.position, q)
        response = super().measure(q, ch)
        if (
            self._finishing
            and response is not None
            and distance >= self.replan_distance
        ):
            self.replans += 1
            raise ReplanAfterBearing()
        return response

    def finish_source(self, ch):
        self._localization_steps[ch] = self._localization_steps.get(ch, 0) + 1
        if self._localization_steps[ch] > 15:
            self.fallback_source(ch)
            return
        self._finishing = True
        try:
            super().finish_source(ch)
        except ReplanAfterBearing:
            pass
        finally:
            self._finishing = False

    def run(self, resume=False):
        result = super().run(resume)
        result["localization_replans"] = self.replans
        return result
