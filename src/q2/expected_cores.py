"""Fine raster for inner union, independent coarse raster for uncertain outer union.
Separating domains prevents distant pending blocks from destroying local resolution.
"""

import json, math
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, label

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "outputs/q2"


def raster(cells, step, bbox, accept=False):
    xmin, ymin, xmax, ymax = bbox
    nx = round((xmax - xmin) / step)
    ny = round((ymax - ymin) / step)
    out = np.zeros((ny, nx), float if accept else bool)
    for a, b, c, e, *_ in cells:
        ix0 = max(0, math.floor((a - xmin) / step + 1e-9))
        ix1 = min(nx, math.ceil((c - xmin) / step - 1e-9))
        iy0 = max(0, math.floor((b - ymin) / step + 1e-9))
        iy1 = min(ny, math.ceil((e - ymin) / step - 1e-9))
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        if accept:
            xx = xmin + np.arange(ix0, ix1) * step
            yy = ymin + np.arange(iy0, iy1) * step
            dx = np.maximum(0, np.minimum(xx + step, c) - np.maximum(xx, a))
            dy = np.maximum(0, np.minimum(yy + step, e) - np.maximum(yy, b))
            out[iy0:iy1, ix0:ix1] += dy[:, None] * dx[None, :]
        else:
            out[iy0:iy1, ix0:ix1] = True
    return out >= step * step - 1e-8 if accept else out


def bounds(cells, step):
    return [
        math.floor(cells[:, 0].min() / step) * step - 2 * step,
        math.floor(cells[:, 1].min() / step) * step - 2 * step,
        math.ceil(cells[:, 2].max() / step) * step + 2 * step,
        math.ceil(cells[:, 3].max() / step) * step + 2 * step,
    ]


def cores(name, step=0.5):
    reg = json.loads((D / (name + "_regions.json")).read_text())
    cells = np.genfromtxt(D / (name + "_cells.csv"), delimiter=",", skip_header=1)
    accepted = cells[cells[:, 5] <= reg["rows"][-1]["threshold_m"]]
    if not len(accepted):
        accepted = np.array(
            [[*reg["best"]["q"], *(np.array(reg["best"]["q"]) + 1), 0, 0]]
        )
    box = bounds(accepted, step)
    while (box[2] - box[0]) * (box[3] - box[1]) / step**2 > 12000000:
        step *= 2
        box = bounds(accepted, step)
    outer_step = 4.0
    obox = bounds(cells, outer_step)
    rows = []
    fields = {}
    for row in reg["rows"]:
        d = row["threshold_m"]
        good = raster(cells[cells[:, 5] <= d], step, box, True)
        outer = raster(cells[cells[:, 4] <= d], outer_step, obox)
        radii = (
            np.maximum(0, distance_transform_edt(good) * step - step / math.sqrt(2))
            * good
        )
        upper = distance_transform_edt(outer) * outer_step + math.sqrt(2) * outer_step
        lab, n = label(good)
        components = []
        for k in range(1, n + 1):
            mask = lab == k
            iy, ix = np.unravel_index(np.where(mask, radii, -1).argmax(), radii.shape)
            components.append(
                {
                    "center_local_m": [
                        box[0] + (ix + 0.5) * step,
                        box[1] + (iy + 0.5) * step,
                    ],
                    "radius_lower_m": float(radii[iy, ix]),
                    "raster_area_m2": float(mask.sum() * step**2),
                }
            )
        # A radius bound at a pixel center is not a bound for every point of that
        # pixel. Subtract its half-diagonal before counting conservative core area.
        rows.append(
            {
                **row,
                "raster_step_m": step,
                "outer_raster_step_m": outer_step,
                "max_radius_lower_m": float(radii.max()),
                "max_radius_upper_m": float(upper[outer].max()) if outer.any() else 0,
                "components": sorted(components, key=lambda z: -z["radius_lower_m"]),
                "core_area_at_r_m2": {
                    str(r): float((radii >= r + step / math.sqrt(2)).sum() * step**2)
                    for r in [1, 2, 5, 10, 20]
                },
            }
        )
        fields["radius_" + str(round(100 * row["epsilon"]))] = np.maximum(
            0, np.nextafter(radii.astype("float32"), np.float32(-np.inf))
        )
    np.savez_compressed(
        D / (name + "_cores.npz"), xmin=box[0], ymin=box[1], step=step, **fields
    )
    (D / (name + "_cores.json")).write_text(
        json.dumps(
            {
                "rows": rows,
                "interpretation": "fine inner raster fully covered by accepted rectangles; outer upper uses independent conservative coarse raster",
                "largest_circle_status": "numerical bounds, not exact maximum",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        name,
        "cores",
        [
            (r["epsilon"], r["max_radius_lower_m"], r["max_radius_upper_m"])
            for r in rows
        ],
        flush=True,
    )
