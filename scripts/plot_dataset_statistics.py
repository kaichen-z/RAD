#!/usr/bin/env python3
"""Render archive-derived RAD statistics (requires matplotlib)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

COLORS = {"normal": "#bdb7d7", "missing": "#8cccc5", "stained": "#f6c57d",
          "scratched": "#ee9588", "squeezed": "#89b5ce"}
DEFECTS = ("missing", "stained", "scratched", "squeezed")


def draw_distribution(ax, statistics):
    keys = ("normal", *DEFECTS)
    totals = statistics["totals"]
    wedges, _, labels = ax.pie(
        [totals[k] for k in keys], colors=[COLORS[k] for k in keys],
        startangle=90, counterclock=False, autopct="%1.1f%%", pctdistance=.78,
        wedgeprops={"width": .44, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 16},
    )
    for label in labels:
        label.set_fontweight("medium")
    ax.text(0, .08, f'{totals["images"]:,}', ha="center", va="center", fontsize=29,
            fontweight="bold")
    ax.text(0, -.17, "RGB images", ha="center", fontsize=15)
    ax.legend(wedges, [f"{k.title()}  {totals[k]:,}" for k in keys],
              loc="lower center", bbox_to_anchor=(.5, -.20), ncol=2,
              frameon=False, fontsize=13, columnspacing=1.2, handlelength=1.0)
    ax.set_title("(b) Image distribution", loc="left", fontsize=20, fontweight="bold", pad=15)


def draw_pixels(ax, statistics):
    categories = list(statistics["categories"])
    x = np.arange(len(categories))
    width = .19
    for i, defect in enumerate(DEFECTS):
        values = [statistics["categories"][c]["mean_anomalous_pixel_percent"][defect]
                  for c in categories]
        valid = np.array([v is not None and v > 0 for v in values])
        ax.bar((x + (i-1.5)*width)[valid], np.array([v or 0 for v in values])[valid],
               width, label=defect.title(), color=COLORS[defect],
               edgecolor="#555555", linewidth=.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(.02, 15)
    ticks = [.03, .1, .3, 1, 3, 10]
    ax.set_yticks(ticks, [str(v) for v in ticks], fontsize=12)
    ax.set_ylabel("Mean anomalous pixels (%)\nFull image area; logarithmic scale", fontsize=13)
    ax.set_xticks(x, [c.title() for c in categories], rotation=35, ha="right", fontsize=12)
    ax.grid(axis="y", which="major", color="#dddddd", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=12)
    ax.set_title("(c) Pixel-wise anomaly coverage", loc="left", fontsize=20,
                 fontweight="bold", pad=14)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statistics", type=Path)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    statistics = json.loads(args.statistics.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    figure = plt.figure(figsize=(13, 10), facecolor="white")
    examples = figure.add_axes([.035, .49, .54, .43])
    examples.imshow(Image.open(args.examples))
    examples.axis("off")
    examples.set_title("(a) Four defect types", loc="left", fontsize=20,
                       fontweight="bold", pad=15)
    distribution = figure.add_axes([.65, .56, .30, .36])
    draw_distribution(distribution, statistics)
    pixels = figure.add_axes([.085, .105, .89, .29])
    draw_pixels(pixels, statistics)
    for extension in ("png", "pdf"):
        figure.savefig(args.output / f"dataset-statistics.{extension}", dpi=220,
                       facecolor="white")
    plt.close(figure)
    # A standalone pair of quantitative panels replaces the legacy statistics PDF.
    figure, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [1, 1]})
    draw_distribution(axes[0], statistics)
    draw_pixels(axes[1], statistics)
    figure.subplots_adjust(hspace=.65, bottom=.15, left=.1, right=.98, top=.94)
    figure.savefig(args.output / "data-statistic.pdf", dpi=220, facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
