"""Replay the retained directional-source scene in the independent local engine."""

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Load the directional engine before the shared robot module search path is added.
from simulator import IndependentQ4Engine
from strategy import make_robot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/q4/replay.json")
    a = p.parse_args()
    scene = json.loads((ROOT / "data/q4/scene.json").read_text())
    mode = scene.get("noise_mode", "blake_uniform")
    if mode == "spatial":
        mode = "blake_uniform"
    engine = IndependentQ4Engine(scene["sources"], scene["noise_seed"], mode)
    policy = make_robot(engine.dispatch)
    result = policy.run()
    evaluation = engine.evaluation()
    assert result["certified_complete"] and evaluation["all_cleared"] and engine.exited
    expected = json.loads((ROOT / "data/q4/replay_reference.json").read_text())
    assert (
        abs(evaluation["virtual_time_s"] - expected["evaluation"]["virtual_time_s"])
        < 1e-6
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            dict(
                scope="local scene; 1770 m support",
                evaluation=evaluation,
                actions=engine.log,
                result=result,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(json.dumps(dict(passed=True, **evaluation), ensure_ascii=False))


if __name__ == "__main__":
    main()
