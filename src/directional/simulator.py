"""Independent scalar Q4 engine for validation, not an official simulator.

No imports from any existing engine, generator, strategy, or noise module.
Implements successful ordinary robot requests, not an HTTP validation server.
Mechanics follow the recorded client evidence; population assumptions are
explicit. Synthetic scene keys and truth stay in the evaluation harness.
"""

import cmath
import hashlib
import math
import random
import time


def node_value(key, channel, ix, iy, mode="blake_uniform"):
    message = f"{key}:{channel}:{ix}:{iy}".encode("ascii")
    if mode == "zero":
        return 0.0
    if mode == "blake_uniform":
        raw = hashlib.blake2b(message, digest_size=8).digest()
    else:
        raw = hashlib.sha256(message).digest()[:8]
    uniform = float(int.from_bytes(raw, "big")) / 2**63 - 1.0
    if mode == "node_extreme":
        return 1.0 if uniform >= 0 else -1.0
    if mode == "node_triangular":
        return math.copysign(1.0 - math.sqrt(1.0 - abs(uniform)), uniform)
    assert mode in ("blake_uniform", "sha_uniform")
    return uniform


def error_value(key, channel, x, y, mode="blake_uniform"):
    ix, iy = math.floor(x / 150.0), math.floor(y / 150.0)
    u, v = x / 150.0 - ix, y / 150.0 - iy
    a, b = 3 * u * u - 2 * u * u * u, 3 * v * v - 2 * v * v * v
    # Independent expanded weight formulation, rather than two successive
    # lerps. Quantized comparisons are checked against the frozen model.
    return math.fsum(
        node_value(key, channel, ix + dx, iy + dy, mode) * wx * wy
        for dx, wx in [(0, 1 - a), (1, a)]
        for dy, wy in [(0, 1 - b), (1, b)]
    )


def quantized_angle(theta, error):
    hundredths = (theta + error) * 100.0
    rounded = int(math.copysign(math.floor(abs(hundredths) + 0.5), hundredths))
    low, high = math.ceil((theta - 1.0) * 100.0), math.floor((theta + 1.0) * 100.0)
    return (min(high, max(low, rounded)) % 36000) / 100.0


def make_independent_scene(label, n, directional, mode="nominal"):
    """Independent MT generator with the same nominal marginal distributions.

    Unlike the original local HMAC-labelled generator, this uses a single
    sequential random stream. This changes hidden scene realization, not the
    policy's prior or inputs. Stress variants intentionally change marginals.
    """
    rng = random.Random(int.from_bytes(hashlib.sha256(label.encode()).digest(), "big"))
    channels = sorted(rng.sample(range(1, 21), n))
    directed = set(rng.sample(channels, directional))
    sources = []
    for channel in channels:
        while True:
            radius = 1770.0 * math.sqrt(rng.random())
            angle = rng.random() * 2.0 * math.pi
            if mode == "boundary_outward":
                radius = 1769.999
            if mode == "cluster":
                radius *= 0.1
            coords = [
                math.copysign(math.floor(abs(v) * 1e6 + 0.5), v) / 1e6
                for v in (radius * math.cos(angle), radius * math.sin(angle))
            ]
            if coords[0] ** 2 + coords[1] ** 2 <= 1770.0**2:
                break
        receive = 1000.0 + rng.randrange(500000001) / 1e6
        orientation = rng.random() * 360.0 if channel in directed else None
        if mode in ("min_range", "boundary_outward"):
            receive = 1000.0
        if mode == "max_range":
            receive = 1500.0
        if mode == "boundary_outward" and orientation is not None:
            orientation = math.degrees(angle) % 360.0
        sources.append(
            dict(
                channel=channel,
                x=coords[0],
                y=coords[1],
                radius=receive,
                orientation=orientation,
                cleared=False,
            )
        )
    return sources, rng.getrandbits(64)


class IndependentQ4Engine:
    def __init__(
        self,
        sources,
        noise_seed,
        noise_mode="blake_uniform",
        robot_id="private-research",
    ):
        self.sources = {s["channel"]: dict(s) for s in sources}
        self.noise_seed, self.noise_mode, self.robot_id = (
            noise_seed,
            noise_mode,
            robot_id,
        )
        self.position, self.channel, self.virtual_us = (0.0, 0.0), 1, 0
        self.entered, self.exited = False, False
        self.log, self.ids = [], set()
        self.counts = dict(
            measure=0, switch=0, clear_success=0, clear_failure=0, distance_m=0.0
        )

    def dispatch(self, path, request):
        assert path in ("/enter", "/measure", "/clear", "/exit")
        expected = {"arena_id", "robot_id", "request_id"}
        if path in ("/measure", "/clear"):
            expected |= {"position", "channel"}
        assert set(request) == expected and request["arena_id"] == "default"
        assert (
            request["robot_id"] == self.robot_id
            and request["request_id"] not in self.ids
        )
        assert not self.exited
        self.ids.add(request["request_id"])
        response = dict(accepted=True, real_timestamp_ms=0)
        if path == "/enter":
            assert not self.entered
            self.entered = True
            response.update(
                max_virtual_duration_s=360000,
                max_real_duration_s=1200,
                remaining_real_duration_s=1200,
            )
        else:
            assert self.entered
            if path == "/exit":
                self.exited = True
                response["exit_reason"] = "user_exit"
            else:
                point = tuple(float(request["position"][a]) for a in ("x", "y"))
                channel = request["channel"]
                assert all(math.isfinite(v) and abs(v) <= 2000000 for v in point)
                assert isinstance(channel, int) and 1 <= channel <= 20
                distance = abs(
                    complex(point[0] - self.position[0], point[1] - self.position[1])
                )
                self.counts["distance_m"] += distance
                self.virtual_us += int(distance * 200000.0 + 0.5)
                self.position = point
                source = self.sources.get(channel)
                active = source is not None and not source["cleared"]
                displacement = (
                    complex(point[0] - source["x"], point[1] - source["y"])
                    if active
                    else None
                )
                radius = abs(displacement) if active else math.inf
                if path == "/measure":
                    switched = int(channel != self.channel)
                    self.channel = channel
                    self.counts["measure"] += 1
                    self.counts["switch"] += switched
                    self.virtual_us += (5 + switched) * 1000000
                    visible = active and radius <= source["radius"]
                    if visible and source["orientation"] is not None:
                        direction = cmath.rect(1.0, math.radians(source["orientation"]))
                        visible = (displacement * direction.conjugate()).real >= 0.0
                    response["measure_result"] = (
                        "no_signal"
                        if not visible
                        else "near" if radius <= 5 else "direction"
                    )
                    if response["measure_result"] == "direction":
                        theta = math.degrees(cmath.phase(-displacement)) % 360.0
                        error = error_value(
                            self.noise_seed, channel, *point, self.noise_mode
                        )
                        response["svd_deg"] = quantized_angle(theta, error)
                else:
                    success = active and radius <= 20.0
                    self.virtual_us += (5 if success else 3) * 1000000
                    self.counts["clear_success" if success else "clear_failure"] += 1
                    response["clear_result"] = (
                        "success" if success else "no_target_in_range"
                    )
                    if success:
                        source["cleared"] = True
        response["virtual_time_s"] = self.virtual_us / 1e6
        assert self.virtual_us < 360000000000
        self.log.append(
            dict(path=path, request=request.copy(), response=response.copy())
        )
        return 200, response

    def evaluation(self):
        n = len(self.sources)
        cleared = sum(s["cleared"] for s in self.sources.values())
        seconds = self.virtual_us / 1e6
        return dict(
            source_count=n,
            cleared_count=cleared,
            all_cleared=cleared == n,
            virtual_time_s=seconds,
            seconds_per_total_source=seconds / n,
            average_time_s=seconds / cleared if cleared else None,
            operation_counts=self.counts.copy(),
            action_count=len(self.log),
            environment="independent_local_engine_not_official",
            noise_mode=self.noise_mode,
        )
