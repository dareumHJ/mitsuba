#!/usr/bin/env python3
"""Figures for contact_sweep.py: geometry schematic and Pout/Pin vs depth."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SERIES = {15.0: "#2a78d6", 30.0: "#eb6834"}     # categorical slots 1, 2
SILICONE = "#1baf7a"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def schematic(path, depths, L=50.0, R=0.5, a=0.12, b=0.125):
    L1 = 0.5 * (L - math.pi * R)
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    th = np.linspace(0, math.pi, 400)
    for r, color, lw in [(b, INK, 1.4), (a, INK2, 0.8)]:
        for sgn in (+1, -1):
            rr = R + sgn * r
            x = R - rr * np.cos(th)
            z = L1 + rr * np.sin(th)
            ax.plot(np.r_[R - rr, x, R + rr], np.r_[L1 - 0.05, z, L1 - 0.05],
                    color=color, lw=lw, ls="-" if r == b else "--")
    z_top = L1 + R + b
    for d in depths:
        z_p = z_top - d
        ax.axhline(z_p, color=SILICONE, lw=0.8, alpha=0.35 + 0.65 * (d / max(depths)))
        ax.text(1.25, z_p, f"d = {d * 1000:.0f} µm", va="center", fontsize=8, color=INK2)
    ax.axhspan(z_top - max(depths), z_top + 0.25, color=SILICONE, alpha=0.08)
    ax.text(0.5, z_top + 0.07, "silicone block (n = 1.41), lowered along −z",
            ha="center", fontsize=9, color=INK)
    ax.text(-0.17, L1 + 0.05, "from input", ha="right", fontsize=8, color=MUTED)
    ax.text(1.17, L1 + 0.05, "to output", ha="left", fontsize=8, color=MUTED)
    ax.set_xlim(-0.55, 1.6)
    ax.set_ylim(L1 + 0.0, z_top + 0.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("Bend side view (xz): silicone contact depth d", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def results(path, rows):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for th, color in SERIES.items():
        rr = [r for r in rows if r["theta_deg"] == th]
        if not rr:
            continue
        d = [r["depth_um"] for r in rr]
        y = [r["pout_pin"] for r in rr]
        ax.plot(d, y, color=color, lw=2, marker="o", ms=7,
                markeredgecolor="#fcfcfb", markeredgewidth=2,
                label=f"launch half-angle {th:.0f}°")
        below = th == 30.0
        ax.annotate(f"{y[0]:.3f}", (d[0], y[0]), xytext=(-10, -12 if below else 8),
                    textcoords="offset points", ha="right", fontsize=8, color=INK2)
        ax.annotate(f"{th:.0f}°: {y[-1]:.3f}", (d[-1], y[-1]), xytext=(10, -7 if below else 7),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    depths = sorted({r["depth_um"] for r in rows})
    area = {r["depth_um"]: r["contact_area_mm2"] for r in rows}
    ax.set_xticks(depths)
    ax.set_xticklabels([f"{d}\n{area[d]:.2f} mm²" for d in depths])
    ax.set_xlabel("contact depth d [µm]  /  silicone-covered cladding area")
    ax.set_ylabel("Pout / Pin")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-45, max(depths) + 60)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("U-bend (R = 0.5 mm): output vs silicone contact depth",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="contact_sweep")
    args = ap.parse_args()
    out = Path(args.dir)
    with open(out / "results.csv", encoding="utf-8") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["depth_um"] = int(r["depth_um"])
    depths_mm = sorted({r["depth_um"] / 1000 for r in rows})
    schematic(out / "geometry_side_view.png", depths_mm)
    results(out / "pout_pin_vs_depth.png", rows)
    print("figures written to", out.resolve())


if __name__ == "__main__":
    main()
