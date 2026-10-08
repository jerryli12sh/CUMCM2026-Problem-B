"""One-command reproduction of geometry, metrics, core regions, and local policies."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help="Rerun all 300 paired Q3 scenes (600 policy runs)",
    )
    p.add_argument(
        "--figures-only",
        action="store_true",
        help="Rebuild static figures from saved data and traces",
    )
    a = p.parse_args()
    started = time.monotonic()
    if a.figures_only:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/figures.py")], check=True, cwd=ROOT
        )
        return
    steps = [
        ["src/q1.py", "--self-test"],
        ["scripts/summarize.py"],
        ["src/q2/evaluate.py"],
        ["src/search/evaluate.py", "--n", "300" if a.full else "3"],
        ["src/directional/evaluate.py"],
        ["scripts/figures.py"],
    ]
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    records = []
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step[0]}", flush=True)
        t = time.monotonic()
        with (out / f"step_{i}.txt").open("w") as log:
            subprocess.run(
                [sys.executable, str(ROOT / step[0]), *step[1:]],
                cwd=ROOT,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        records.append(
            dict(step=step, passed=True, seconds=round(time.monotonic() - t, 3))
        )
    result = dict(
        passed=True,
        mode="full" if a.full else "quick",
        steps=records,
        seconds=round(time.monotonic() - started, 3),
    )
    (out / "reproduction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f'All {len(steps)} steps passed in {result["seconds"]:.1f} seconds. Results: results/; detailed run output: outputs/'
    )


if __name__ == "__main__":
    main()
