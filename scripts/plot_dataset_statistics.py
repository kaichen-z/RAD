#!/usr/bin/env python3
"""Render RAD's original three-panel statistics layout using verified counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

DEFECTS = ("missing", "stained", "scratched", "squeezed")
COLORS = {"normal": "#b9b3d2", "missing": "#87c5bd", "stained": "#edc486",
          "scratched": "#e69a8c", "squeezed": "#86b2cc"}
LABELS = ("Binder", "Bowl", "Box", "Can", "Charger", "Cup1", "Cup2", "Glue",
          "Phone", "Duck", "Spoon", "Spray", "Tennis")
INK = "#252b30"


def draw_examples(figure, image):
    panels = [
        ("Stained", (45, 40, 415, 350), (.02, .56, .18, .31)),
        ("Scratched", (515, 40, 875, 350), (.235, .56, .18, .31)),
        ("Missing", (45, 510, 415, 823), (.02, .16, .18, .31)),
        ("Squeezed", (515, 510, 875, 823), (.235, .16, .18, .31)),
    ]
    for label, box, bounds in panels:
        ax = figure.add_axes(bounds)
        ax.imshow(image.crop(box))
        ax.axis("off")
        ax.text(.5, -.04, label, transform=ax.transAxes, ha="center", va="top",
                color=INK, fontsize=12)
    for x, y in [([.217, .217], [.12, .91]), ([.015, .425], [.515, .515])]:
        figure.add_artist(Line2D(x, y, transform=figure.transFigure,
                                color="#b3b6b8", linewidth=.7, dashes=(4, 5)))
    figure.text(.015, .962, "(a) Four defect types", fontsize=13, weight="bold", color=INK)


def draw_distribution(figure, statistics):
    keys = ("normal", *DEFECTS)
    totals = statistics["totals"]
    ax = figure.add_axes([.605, .555, .28, .365])
    ax.pie([totals[k] for k in keys], colors=[COLORS[k] for k in keys],
           labels=[k.title() for k in keys], startangle=90, counterclock=False,
           autopct="%1.1f%%", pctdistance=.7, labeldistance=1.12,
           wedgeprops={"edgecolor": "white", "linewidth": 1.1},
           textprops={"fontsize": 10.5, "color": INK})
    figure.text(.485, .962, f"(b) Image distribution (n = {totals['images']:,})",
                fontsize=12, weight="bold", color=INK)


def draw_coverage(figure, statistics):
    categories = list(statistics["categories"])
    ax = figure.add_axes([.52, .16, .455, .285])
    x = np.arange(len(categories))
    width = .19
    short = ("Mi.", "St.", "Sc.", "Sq.")
    for index, defect in enumerate(DEFECTS):
        values = [statistics["categories"][c]["mean_anomalous_pixel_percent"][defect]
                  for c in categories]
        valid = np.array([value is not None and value > 0 for value in values])
        heights = np.array([value or 0 for value in values])
        ax.bar((x+(index-1.5)*width)[valid], heights[valid], width,
               color=COLORS[defect], label=short[index], edgecolor="#777777", linewidth=.35,
               zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(.02, 15)
    ticks = [.03, .1, .3, 1, 3, 10]
    ax.set_yticks(ticks, [str(tick) for tick in ticks], fontsize=8.8)
    ax.set_ylabel("Anomalous pixels (%)", fontsize=10, labelpad=3)
    ax.set_xticks(x, LABELS, rotation=45, ha="right", fontsize=9.2)
    ax.tick_params(length=2.5, width=.5, colors=INK, pad=2)
    ax.grid(axis="y", which="major", color="#dedfe1", linewidth=.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(.6)
    ax.legend(ncol=4, frameon=False, fontsize=9.3, handlelength=1.1,
              handletextpad=.4, columnspacing=1.1, loc="lower left", bbox_to_anchor=(0, 1.01),
              borderaxespad=0)
    figure.text(.485, .52, "(c) Pixel-wise anomaly coverage", fontsize=13,
                weight="bold", color=INK)
    figure.text(.52, .035, "Mean over images; full-image area; logarithmic axis",
                fontsize=9, color="#626b72")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("statistics", type=Path)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    statistics = json.loads(args.statistics.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "axes.unicode_minus": False})
    figure = plt.figure(figsize=(10.0, 6.0), facecolor="white")
    draw_examples(figure, Image.open(args.examples))
    draw_distribution(figure, statistics)
    draw_coverage(figure, statistics)
    for extension in ("png", "pdf"):
        figure.savefig(args.output/f"dataset-statistics.{extension}", dpi=300,
                       facecolor="white", bbox_inches="tight", pad_inches=.05)
    figure.savefig(args.output/"data-statistic.pdf", dpi=300,
                   facecolor="white", bbox_inches="tight", pad_inches=.05)
    plt.close(figure)


if __name__ == "__main__":
    main()
