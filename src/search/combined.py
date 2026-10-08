"""Earlier second bearing, followed by bounded global replanning."""

from events import EventRobot
from approach import ApproachRobot


class CombinedRobot(EventRobot, ApproachRobot):
    pass
