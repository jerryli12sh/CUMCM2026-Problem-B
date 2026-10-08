"""Run paired local scenes. The policy sees only action responses."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from geometry import contains
from routing import RoutingRobot
from short_baseline import LeanGradientRobot
from scene import RecoveredPracticeSimulator

ROOT = Path(__file__).resolve().parents[2]


def episode(index, method, trace=False):
    # Hashing here defines reproducible random streams, not file bookkeeping.
    seed = hashlib.sha256(f"recovered-independent-{index}".encode()).digest()
    simulator = RecoveredPracticeSimulator(seed)
    config = json.loads((Path(__file__).parent / "settings.json").read_text())
    options = dict(mode="joint_smart")
    if method == "short_baseline":
        options.update(
            config["route_settings"],
            ring_radius=config["config"]["ring"],
            try_radius=config["config"]["try_radius"],
            discovery_gain=config["config"]["discovery_gain"],
        )

    def audit(robot, channel, action, safe):
        source = simulator._sources.get(channel)
        belief = robot.beliefs[channel]
        if source and belief.known and not source.cleared:
            assert contains(belief.poly, (source.x, source.y))
            assert all(
                math.dist((source.x, source.y), q) > 20 - 1e-7 for q in belief.failed
            )
        if action == "clear" and safe:
            assert (
                source and math.dist(robot.position, (source.x, source.y)) <= 20 + 1e-7
            )

    cls = RoutingRobot if method == "baseline" else LeanGradientRobot
    robot = cls(simulator.dispatch, audit=audit, **options)
    result = robot.run()
    evaluation = simulator.evaluation()
    assert (
        result["certified_complete"] and evaluation["all_cleared"] and simulator.exited
    )
    row = dict(
        index=index,
        method=method,
        **{
            k: evaluation[k]
            for k in [
                "source_count",
                "cleared_count",
                "all_cleared",
                "virtual_time_s",
                "average_time_s",
            ]
        },
        **evaluation["operation_counts"],
    )
    if trace:
        row["sources"] = [
            dict(channel=s.channel, x=s.x, y=s.y, radius=s.radius)
            for s in simulator._sources.values()
        ]
        row["actions"] = simulator.log
        row["discovery_stations"] = robot.discovery_stations
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--start", type=int, default=30000)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/q3")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = {
        (int(r["index"]), r["method"]): r
        for r in csv.DictReader((ROOT / "data/q3/paired_300.csv").open())
    }
    rows = []
    for index in range(args.start, args.start + args.n):
        for method in ["baseline", "short_baseline"]:
            row = episode(index, method)
            if (index, method) in reference:
                old = reference[index, method]
                assert (
                    abs(row["virtual_time_s"] - float(old["virtual_time_s"])) < 1e-6
                ), (index, method)
            rows.append(row)
        if (index - args.start + 1) % 10 == 0:
            print("Paired scenes completed:", index - args.start + 1, flush=True)
    with (args.output / "episodes.csv").open("w") as f:
        writer = csv.DictWriter(f, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    selection = json.loads((ROOT / "data/q3/replay_selection.json").read_text())[
        "index"
    ]
    for method in ["baseline", "short_baseline"]:
        (args.output / f"trace_{method}.json").write_text(
            json.dumps(episode(selection, method, True), indent=2)
        )
    print(
        json.dumps(
            dict(
                passed=True,
                paired_scenes=args.n,
                scope="local synthetic scenes",
                output=str(args.output),
            )
        )
    )


if __name__ == "__main__":
    main()
