"""Verify every retained 20-station layout on the 1770 m support disk."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/directional"))
from certify_layout_integer import certify

if __name__ == "__main__":
    catalog = json.loads((ROOT / "src/directional/catalog_lattice20.json").read_text())
    rows = []
    for item in catalog:
        proof = certify(
            [[round(v * 1000) for v in p] for p in item["points"]], domain_mm=1770000
        )
        assert proof["certified"]
        rows.append(
            dict(
                layout=item["degrees"],
                stations=len(item["points"]),
                checked=proof["checked"],
                leaves=proof["leaf_count"],
            )
        )
    result = dict(
        passed=True,
        layouts=len(rows),
        domain_m=1770,
        minimum_range_m=1000,
        arithmetic="integer millimetres",
        details=rows,
    )
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    (out / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{len(rows)} layouts passed integer coverage verification on the 1770 m support disk."
    )
