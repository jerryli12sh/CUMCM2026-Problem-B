"""Build repository figures from retained numerical results and compact traces."""

import csv
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "results/figures"
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#354453",
        "text.color": "#172b3a",
        "axes.titleweight": "bold",
        "axes.edgecolor": "#b3bcc5",
        "xtick.color": "#60717e",
        "ytick.color": "#60717e",
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "font.size": 10,
    }
)
BLUE = "#1d607c"
RED = "#c86445"
TEAL = "#27887b"
GRAY = "#9aadb8"


def save(fig, name):
    fig.savefig(DEST / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def q1():
    p = np.array([[-18, -6 * np.sqrt(3)], [18, -6 * np.sqrt(3)], [0, 12 * np.sqrt(3)]])
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.fill(*p.T, color=BLUE, alpha=0.12)
    ax.plot(*np.vstack([p, p[0]]).T, color=BLUE, lw=2)
    for r, c, ls, label in [
        (20, RED, "--", "Optical radius: 20 m"),
        (12 * np.sqrt(3), TEAL, "-", "Required radius: 20.785 m"),
    ]:
        ax.add_patch(Circle((0, 0), r, fill=False, color=c, ls=ls, label=label))
    ax.scatter(*p.T, color=BLUE, zorder=3)
    ax.scatter(0, 0, c="black", s=16)
    ax.set(
        xlim=(-30, 30),
        ylim=(-28, 30),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="Q1  |  Diameter and one-shot clearance",
    )
    ax.text(0, -25, "Triangle diameter = 36 m", ha="center", fontsize=10)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    save(fig, "q1_geometry.png")


def q2():
    cells = np.genfromtxt(
        ROOT / "data/q2/standard_taper_cells.csv", delimiter=",", skip_header=1
    )
    rows = json.loads((ROOT / "data/q2/standard_taper_cores.json").read_text())["rows"]
    row = rows[2]
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.3, 1]}
    )
    ax = axes[0]
    for idx, color in [
        (4, "#e8f0f3"),
        (3, "#b3d3de"),
        (2, "#63a7ba"),
        (1, "#367f99"),
        (0, "#194b69"),
    ]:
        accepted = cells[cells[:, 5] <= rows[idx]["threshold_m"]]
        patches = [Rectangle((a, b), c - a, d - b) for a, b, c, d, *_ in accepted]
        ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor="none"))
    for r in row["components"]:
        ax.add_patch(
            Circle(r["center_local_m"], 80.16, fill=False, edgecolor=RED, lw=1.7)
        )
    ax.scatter([907.4287] * 2, [577.0756, -577.0756], color="#152c3f", s=18)
    ax.set(
        xlim=(550, 1200),
        ylim=(300, 950),
        xlabel="Second station x (m)",
        ylabel="Second station y (m)",
        title="Second-station region (upper half)",
        aspect="equal",
    )
    ax.text(
        1170, 810, "5% safe circles\nr = 80.16 m", color=RED, ha="right", fontsize=10
    )
    ax.grid(alpha=0.1)
    eps = np.array([r["epsilon"] for r in rows]) * 100
    radii = [r["max_radius_lower_m"] for r in rows]
    axes[1].plot(eps, radii, "o-", color=BLUE, lw=2)
    axes[1].scatter([5], [row["max_radius_lower_m"]], color=RED, s=60, zorder=4)
    for e, r in zip(eps, radii):
        axes[1].annotate(
            f"{r:.1f} m", (e, r), xytext=(3, 8), textcoords="offset points", fontsize=9
        )
    axes[1].set(
        xlabel="Expected-diameter allowance (%)",
        ylabel="Safe-circle radius (m)",
        title="More room to choose a measurement site",
        ylim=(0, 235),
    )
    axes[1].grid(alpha=0.15)
    fig.tight_layout(w_pad=3)
    save(fig, "q2_candidates.png")


def q3():
    rr = list(csv.DictReader((ROOT / "data/q3/paired_300.csv").open()))
    groups = {
        m: sorted([r for r in rr if r["method"] == m], key=lambda r: int(r["index"]))
        for m in ["baseline", "short_baseline"]
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    labels = ["Travel", "Measure", "Switch", "Clear"]
    colors = [BLUE, "#63a7ba", "#c4d9e3", RED]
    left = np.zeros(2)
    for label, color, keys, factors in zip(
        labels,
        colors,
        [["distance_m"], ["measure"], ["switch"], ["clear_success", "clear_failure"]],
        [[0.2], [5], [1], [5, 3]],
    ):
        x = [
            np.mean(
                [
                    sum(float(r[k]) * f for k, f in zip(keys, factors))
                    / int(r["source_count"])
                    for r in g
                ]
            )
            for g in groups.values()
        ]
        axes[0].barh([1, 0], x, left=left, color=color, label=label, height=0.42)
        left += x
    axes[0].set_yticks([1, 0], ["Intersection baseline", "Short-baseline policy"])
    axes[0].set(
        xlabel="Mean task time (s/source)",
        xlim=(0, 272),
        title="300 paired local scenes",
    )
    for y, x in zip([1, 0], left):
        axes[0].text(x + 3, y, f"{x:.2f}", va="center", fontsize=10)
    axes[0].legend(
        ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(-0.15, -0.4)
    )
    delta = np.array(
        [
            float(a["average_time_s"]) - float(b["average_time_s"])
            for a, b in zip(*groups.values())
        ]
    )
    axes[1].hist(delta, bins=25, color=TEAL, alpha=0.9, edgecolor="white")
    axes[1].set_ylim(0, 39)
    axes[1].axvline(0, color=GRAY, lw=1)
    axes[1].axvline(delta.mean(), color=RED, lw=1.8)
    axes[1].set(
        xlabel="Time saved per scene (s/source)",
        ylabel="Scenes",
        title="7.17% lower mean time",
    )
    axes[1].text(
        0.98,
        0.94,
        "281 / 300 scenes faster\nMean saving: 16.82 s/source",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(w_pad=3)
    save(fig, "q3_results.png")


def routes():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    names = [
        ("q3", "baseline", "Q3  |  Intersection baseline"),
        ("q3", "short_baseline", "Q3  |  Short baseline"),
        ("q4", "directional", "Q4  |  Directional-source policy"),
    ]
    for ax, (q, m, title) in zip(axes, names):
        path = ROOT / f"data/{q}/trace_{m}.csv"
        r = list(csv.DictReader(path.open()))
        xy = np.array([[float(z["x"]), float(z["y"])] for z in r])
        s = list(csv.DictReader((ROOT / f"data/{q}/sources_{m}.csv").open()))
        sx = np.array([[float(z["x"]), float(z["y"])] for z in s])
        ax.add_patch(Circle((0, 0), 1800, fill=False, ec=GRAY, ls="--", lw=0.8))
        ax.plot(*xy.T, color=BLUE, lw=0.8, alpha=0.8)
        ax.scatter(*sx.T, marker="*", s=38, color=RED, zorder=5)
        ax.scatter(0, 0, s=25, color=TEAL, zorder=6)
        ax.set(
            aspect="equal",
            xlim=(-2050, 2050),
            ylim=(-2050, 2050),
            title=title,
            xlabel="x (m)",
        )
        ax.set_xticks([-1500, 0, 1500])
        ax.set_yticks([-1500, 0, 1500])
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("y (m)")
    fig.text(
        0.5,
        0.03,
        "Replayed local scenes. Stars: sources   |   Green point: start   |   Blue line: executed path",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.2, wspace=0.3)
    save(fig, "routes.png")


def formal():
    r = list(csv.DictReader((ROOT / "data/formal_results.csv").open()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, q, color in zip(axes, ["3", "4"], [BLUE, TEAL]):
        rows = [z for z in r if z["question"] == q]
        v = [float(z["seconds_per_source"]) for z in rows]
        ax.bar([1, 2, 3], v, color=color, width=0.5)
        ax.axhline(np.mean(v), color=RED, ls="--", label=f"Case mean {np.mean(v):.2f}")
        for i, z in enumerate(rows):
            ax.text(
                i + 1,
                v[i] + 9,
                f'{v[i]:.2f}\n{z["cleared_sources"]} sources',
                ha="center",
                fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            )
        ax.set(
            xlabel="Formal test case",
            ylabel="Task time (s/source)",
            title=f"Q{q}  |  Three formal tests",
            ylim=(0, max(v) * 1.3),
            xticks=[1, 2, 3],
        )
        ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(w_pad=3)
    save(fig, "formal_results.png")


if __name__ == "__main__":
    for f in [q1, q2, q3, routes, formal]:
        f()
    print("Five figures written to results/figures")
