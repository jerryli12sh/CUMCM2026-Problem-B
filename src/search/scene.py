"""Deterministic local Q3 scene and fixed spatial-error model for paired evaluation."""

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import hmac
import math
from pathlib import Path
from simulator import Simulator, Source, serve


def round_away(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


class CounterSource:
    def __init__(self, seed):
        if len(seed) != 32:
            raise ValueError("Generation seed must contain 32 bytes")
        self.seed = seed
        self.counters = defaultdict(int)

    def next(self, label):
        count = self.counters[label]
        self.counters[label] += 1
        message = (
            b"practice-case-v1\0" + label.encode() + b"\0" + count.to_bytes(8, "big")
        )
        return int.from_bytes(
            hmac.new(self.seed, message, hashlib.sha256).digest()[:8], "big"
        )

    def uintn(self, label, n):
        if not 1 <= n <= 2**63:
            raise ValueError("Invalid random range")
        threshold = 2**64 % n
        while True:
            value = self.next(label)
            if value >= threshold:
                return value % n

    def float01(self, label):
        return (self.next(label) >> 11) * 2**-53

    def shuffle(self, label, values):
        for i in range(len(values) - 1, 0, -1):
            j = self.uintn(label, i + 1)
            values[i], values[j] = values[j], values[i]


def generate_practice(seed):
    rng = CounterSource(seed)
    n = 10 + rng.uintn("count", 7)
    channels = list(range(1, 21))
    rng.shuffle("channels", channels)
    sources = []
    for ch in sorted(channels[:n]):
        while True:
            radius = 1770000000 * math.sqrt(rng.float01(f"jammer/{ch}/radius"))
            theta = 2 * math.pi * rng.float01(f"jammer/{ch}/theta")
            x, y = round_away(radius * math.cos(theta)), round_away(
                radius * math.sin(theta)
            )
            if x * x + y * y <= 1770000000**2:
                break
        receive = 1000000000 + rng.uintn(f"jammer/{ch}/receive", 500000001)
        sources.append(Source(ch, x / 1e6, y / 1e6, receive / 1e6))
    return sources, rng.next("noise-seed")


def uint64_float_go(value):
    if value < 2**63:
        return float(value)
    return float((value >> 1) | (value & 1)) * 2


@lru_cache(maxsize=100000)
def grid(seed, channel, ix, iy):
    message = f"{seed}:{channel}:{ix}:{iy}".encode()
    integer = int.from_bytes(hashlib.blake2b(message, digest_size=8).digest(), "big")
    return uint64_float_go(integer) / 2**64 * 2 - 1


def spatial_error(seed, channel, x, y):
    xx, yy = x / 150, y / 150
    ix, iy = math.floor(xx), math.floor(yy)
    ux, uy = max(0.0, min(1.0, xx - ix)), max(0.0, min(1.0, yy - iy))
    tx, ty = ux * ux * (3 - 2 * ux), uy * uy * (3 - 2 * uy)
    a, b = grid(seed, channel, ix, iy), grid(seed, channel, ix + 1, iy)
    c, d = grid(seed, channel, ix, iy + 1), grid(seed, channel, ix + 1, iy + 1)
    lo, hi = a + (b - a) * tx, c + (d - c) * tx
    return lo + (hi - lo) * ty


def quantize_bearing(theta, error):
    value = round_away((theta + error) * 100)
    value = max(math.ceil((theta - 1) * 100), min(math.floor((theta + 1) * 100), value))
    return (value % 36000) / 100


class RecoveredPracticeSimulator(Simulator):
    def __init__(self, seed, robot_id="local-research"):
        sources, noise_seed = generate_practice(seed)
        self.generation_seed = seed.hex()
        super().__init__(sources, seed=noise_seed, robot_id=robot_id)

    def error(self, channel, x, y):
        return spatial_error(self._seed, channel, x, y)

    def response(self, accepted, **fields):
        if accepted and fields.get("measure_result") == "direction":
            s = self._sources[self.channel]
            x, y = self.position
            theta = math.degrees(math.atan2(s.y - y, s.x - x)) % 360
            fields["svd_deg"] = quantize_bearing(theta, self.error(self.channel, x, y))
        return super().response(accepted, **fields)

    def evaluation(self):
        value = super().evaluation()
        value["environment"] = "local_recovered_practice_mechanics_not_official"
        value["generation_seed_hex"] = self.generation_seed
        value["validation_limit"] = (
            "No known-seed official differential test yet; protocol base remains research simulator"
        )
        return value


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed-hex", default="00" * 32)
    p.add_argument("--port", type=int, default=2027)
    p.add_argument(
        "--report", type=Path, default=Path("iterations/recovered_local_episode.json")
    )
    args = p.parse_args()
    seed = bytes.fromhex(args.seed_hex)
    sim = RecoveredPracticeSimulator(seed)
    print(
        "Local reconstructed mechanics; synthetic seed; not an official score.",
        flush=True,
    )
    serve(sim, args.port, args.report)
