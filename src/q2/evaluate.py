"""Evaluate the recommended point and rebuild the 5% candidate-circle result."""

import json
import shutil
from pathlib import Path
from expectation import Geometry, samples, point
from expected_cores import cores

ROOT = Path(__file__).resolve().parents[2]


def main():
    dest = ROOT / "outputs/q2"
    dest.mkdir(parents=True, exist_ok=True)
    name = "standard_taper"
    for suffix in ["_cells.csv", "_regions.json"]:
        shutil.copy2(ROOT / "data/q2" / f"{name}{suffix}", dest / f"{name}{suffix}")
    cores(name)
    actual = json.loads((dest / f"{name}_cores.json").read_text())
    reference = json.loads((ROOT / "data/q2" / f"{name}_cores.json").read_text())
    for a, b in zip(actual["rows"], reference["rows"]):
        assert abs(a["max_radius_lower_m"] - b["max_radius_lower_m"]) < 1e-9
    regions = json.loads((dest / f"{name}_regions.json").read_text())
    g = Geometry()
    x = samples(g, 131072, 116018, "taper")[0]
    result = point(g, regions["best"]["q"], x, 32768)
    assert result["lo"] < 50.3 and result["hi"] > 50.0
    (dest / "point_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=True,
                point_interval=[result["lo"], result["hi"]],
                safe_radius=actual["rows"][2]["max_radius_lower_m"],
            )
        )
    )


if __name__ == "__main__":
    main()
