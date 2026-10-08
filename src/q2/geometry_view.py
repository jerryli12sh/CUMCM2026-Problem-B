"""Use a coarser valid circular enclosure for normal branches only.
The state, exact feedback, station domain, and near/silent bounds remain Q4's.
"""

from geometry import Geometry


class NormalGeometryView:
    def __init__(self, base, arc=12):
        self.base = base
        normal = Geometry(base.S, base.theta, arc=arc)
        self.outer = normal.outer
        self.inner = normal.inner
        self.normal_arc = arc

    def __getattr__(self, name):
        return getattr(self.base, name)
