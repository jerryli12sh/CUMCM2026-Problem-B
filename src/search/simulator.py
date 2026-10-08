"""Local research simulator implementing the four published robot actions.

This is NOT the official simulator. Scenario distributions and spatial noise fields
are explicit test assumptions. Hidden truth is never returned by the action API.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import random
import threading
import time
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Source:
    channel: int
    x: float
    y: float
    radius: float
    orientation: float | None = None
    cleared: bool = False


def generate(seed, n=None, layout="uniform", radius_mode="uniform"):
    rng = random.Random(seed)
    n = rng.randint(10, 16) if n is None else n
    channels = rng.sample(range(1, 21), n)
    cluster_centers = [
        (rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)) for _ in range(3)
    ]
    result = []
    for i, c in enumerate(channels):
        if layout == "clusters":
            cx, cy = cluster_centers[i % 3]
            x, y = cx + rng.gauss(0, 170), cy + rng.gauss(0, 170)
            if math.hypot(x, y) > 1799:
                f = 1799 / math.hypot(x, y)
                x, y = x * f, y * f
        else:
            theta = rng.random() * 2 * math.pi
            r = (
                1800 * math.sqrt(rng.random())
                if layout == "uniform"
                else rng.uniform(1700, 1799.9)
            )
            x, y = r * math.cos(theta), r * math.sin(theta)
        radius = (
            rng.uniform(1000, 1500) if radius_mode == "uniform" else float(radius_mode)
        )
        result.append(Source(c, x, y, radius))
    return result


class Simulator:
    def __init__(self, sources, seed=0, noise="hash", robot_id="local-research"):
        self._sources = {s.channel: s for s in sources}
        self._seed = seed
        self._noise = noise
        self.robot_id = robot_id
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual_us = 0
        self.entered = False
        self.exited = False
        self.started = None
        self.cache = {}
        self.log = []
        self.counts = {
            "measure": 0,
            "switch": 0,
            "clear_success": 0,
            "clear_failure": 0,
            "distance_m": 0.0,
        }
        self.lock = threading.Lock()
        self.error_overrides = {}

    def error(self, channel, x, y):
        x = 0.0 if x == 0 else x
        y = 0.0 if y == 0 else y
        if (channel, x, y) in self.error_overrides:
            return self.error_overrides[channel, x, y]
        if self._noise == "zero":
            return 0.0
        if self._noise == "smooth":
            return math.sin(0.008 * x + 0.017 * channel + self._seed) * math.cos(
                0.007 * y - 0.031 * channel
            )
        key = f"{self._seed}:{channel}:{float(x).hex()}:{float(y).hex()}".encode()
        h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
        value = 2 * (h / (2**64 - 1)) - 1
        return (1.0 if value >= 0 else -1.0) if self._noise == "extreme" else value

    def response(self, accepted, **fields):
        return dict(
            accepted=accepted,
            real_timestamp_ms=int(time.time() * 1000),
            virtual_time_s=self.virtual_us / 1e6 if accepted else 0,
            **fields,
        )

    @staticmethod
    def identifier(value, limit):
        return (
            isinstance(value, str)
            and 1 <= len(value.encode("utf-8")) <= limit
            and not any(unicodedata.category(c) in ("Cc", "Cf") for c in value)
        )

    def dispatch(self, path, data):
        if not self.lock.acquire(blocking=False):
            return 409, self.response(False)
        try:
            return self._dispatch(path, data)
        finally:
            self.lock.release()

    def _dispatch(self, path, data):
        if path not in ("/enter", "/measure", "/clear", "/exit"):
            return 404, self.response(False)
        if not isinstance(data, dict):
            return 400, self.response(False)
        required = {"arena_id", "robot_id", "request_id"}
        if path in ("/measure", "/clear"):
            required |= {"position", "channel"}
        if not required <= data.keys():
            return 400, self.response(False)
        if (
            not isinstance(data["arena_id"], str)
            or not self.identifier(data["robot_id"], 64)
            or not self.identifier(data["request_id"], 128)
        ):
            return 400, self.response(False)
        if path in ("/measure", "/clear"):
            p = data["position"]
            c = data["channel"]
            if not isinstance(p, dict) or not {"x", "y"} <= p.keys():
                return 400, self.response(False)
            if any(
                isinstance(p[a], bool)
                or not isinstance(p[a], (float, int))
                or not math.isfinite(p[a])
                or abs(p[a]) > 2000000
                for a in ("x", "y")
            ):
                return 400, self.response(False)
            if (
                isinstance(c, bool)
                or not isinstance(c, (float, int))
                or not math.isfinite(c)
                or c != int(c)
                or not 1 <= c <= 20
            ):
                return 400, self.response(False)
            if set(p) != {"x", "y"}:
                return 200, self.response(False)
        if (
            set(data) != required
            or data["arena_id"] != "default"
            or data["robot_id"] != self.robot_id
        ):
            return 200, self.response(False)
        rid = data["request_id"]
        signature = (path, json.dumps(data, sort_keys=True, separators=(",", ":")))
        if rid in self.cache:
            old_sig, old_response = self.cache[rid]
            return (
                (200, old_response.copy())
                if old_sig == signature
                else (409, self.response(False))
            )
        if (
            self.exited
            or (path == "/enter" and self.entered)
            or (path != "/enter" and not self.entered)
        ):
            return 200, self.response(False)
        if self.started is not None and time.monotonic() - self.started > 1200:
            self.exited = True
            return 200, self.response(False)
        if path == "/enter":
            self.entered = True
            self.started = time.monotonic()
            res = self.response(
                True,
                max_virtual_duration_s=360000,
                max_real_duration_s=1200,
                remaining_real_duration_s=1200,
            )
        elif path == "/exit":
            self.exited = True
            res = self.response(True, exit_reason="user_exit")
        else:
            q = (float(data["position"]["x"]), float(data["position"]["y"]))
            ch = int(data["channel"])
            distance = math.dist(q, self.position)
            self.position = q
            self.counts["distance_m"] += distance
            self.virtual_us += round(distance / 5 * 1e6)
            source = self._sources.get(ch)
            active = source is not None and not source.cleared
            r = math.hypot(q[0] - source.x, q[1] - source.y) if active else math.inf
            if path == "/measure":
                switched = int(self.channel != ch)
                self.channel = ch
                self.virtual_us += (5 + switched) * 1000000
                self.counts["measure"] += 1
                self.counts["switch"] += switched
                visible = active and r <= source.radius
                if visible and source.orientation is not None:
                    theta = math.radians(source.orientation)
                    visible = (
                        math.cos(theta) * (q[0] - source.x)
                        + math.sin(theta) * (q[1] - source.y)
                        >= -1e-10
                    )
                if not visible:
                    res = self.response(True, measure_result="no_signal")
                elif r <= 5:
                    res = self.response(True, measure_result="near")
                else:
                    theta = math.degrees(math.atan2(source.y - q[1], source.x - q[0]))
                    value = round((theta + self.error(ch, *q)) % 360, 2) % 360
                    res = self.response(True, measure_result="direction", svd_deg=value)
            else:
                success = active and r <= 20
                self.virtual_us += (5 if success else 3) * 1000000
                self.counts["clear_success" if success else "clear_failure"] += 1
                if success:
                    source.cleared = True
                res = self.response(
                    True, clear_result="success" if success else "no_target_in_range"
                )
            if self.virtual_us > 360000000000:
                self.exited = True
        self.cache[rid] = (signature, res.copy())
        self.log.append({"path": path, "request": data.copy(), "response": res.copy()})
        return 200, res

    def evaluation(self):
        n = len(self._sources)
        cleared = sum(s.cleared for s in self._sources.values())
        t = self.virtual_us / 1e6
        return dict(
            source_count=n,
            cleared_count=cleared,
            clear_fraction=cleared / n,
            all_cleared=cleared == n,
            virtual_time_s=t,
            average_time_s=t / cleared if cleared else None,
            operation_counts=self.counts.copy(),
            action_count=len(self.log),
            environment="local_synthetic_not_official",
        )


def serve(sim, port=2027, report_path=None, run=True):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, obj):
            body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(
                405 if self.path in ("/enter", "/exit", "/measure", "/clear") else 404,
                sim.response(False),
            )

        def do_POST(self):
            ctype = self.headers.get("Content-Type", "").lower().replace(" ", "")
            if (
                ctype not in ("application/json", "application/json;charset=utf-8")
                or self.headers.get("Content-Encoding", "identity").lower()
                != "identity"
            ):
                self.reply(415, sim.response(False))
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.reply(400, sim.response(False))
                return
            if size > 65536:
                self.reply(413, sim.response(False))
                return
            try:

                def pairs(values):
                    d = {}
                    for k, v in values:
                        if k in d:
                            raise ValueError("duplicate key")
                        d[k] = v
                    return d

                data = json.loads(
                    self.rfile.read(size).decode("utf-8"),
                    object_pairs_hook=pairs,
                    parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)),
                )

                def depth(v):
                    items = (
                        v.values()
                        if isinstance(v, dict)
                        else v if isinstance(v, list) else []
                    )
                    return 1 + max((depth(x) for x in items), default=0)

                if depth(data) > 17:
                    raise ValueError("deep JSON")
            except (ValueError, UnicodeError, RecursionError):
                self.reply(400, sim.response(False))
                return
            status, res = sim.dispatch(self.path, data)
            self.reply(status, res)
            if self.path == "/exit" and res.get("accepted") and report_path:
                from pathlib import Path

                Path(report_path).write_text(
                    json.dumps(
                        {"result": sim.evaluation(), "actions": sim.log}, indent=2
                    )
                    + "\n"
                )
                print(json.dumps(sim.evaluation()), flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if not run:
        return server
    print(
        f"Local research simulator http://127.0.0.1:{server.server_port}; robot_id={sim.robot_id}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--port", type=int, default=2027)
    p.add_argument("--noise", default="hash")
    p.add_argument("--report")
    args = p.parse_args()
    serve(
        Simulator(generate(args.seed), seed=args.seed, noise=args.noise),
        args.port,
        args.report,
    )
