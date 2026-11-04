#!/usr/bin/env python3
"""Draw the traffic-volume chart (with the peak highlighted) from a bins CSV.

Example:
    python src/plots.py --bins outputs/cam1_bins.csv --out outputs/cam1_peak_chart.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed (works on Colab and servers)
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", required=True, help="bins CSV from aggregate.py")
    ap.add_argument("--out", required=True, help="PNG file to write")
    ap.add_argument("--title", default="Vehicle volume per time bin")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.bins)
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["#c0392b" if i == df["total"].idxmax() and df["total"].max() > 0 else "#4a90d9" for i in x]
    ax.bar(x, df["total"], color=colors, label="Total")
    for direction, style in (("A_to_B", "o-"), ("B_to_A", "s--")):
        if df[direction].sum() > 0:
            ax.plot(x, df[direction], style, color="#2c3e50", alpha=0.7, label=direction)

    step = max(1, len(df) // 12)  # avoid a crowded x axis
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels(df["bin_start"][::step], rotation=45, ha="right")
    ax.set_xlabel("Time bin start")
    ax.set_ylabel("Vehicles counted")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if df["total"].max() > 0:
        peak = df.loc[df["total"].idxmax()]
        ax.annotate(f"Peak: {peak['bin_start']}\n{int(peak['total'])} vehicle(s)",
                    xy=(df["total"].idxmax(), peak["total"]), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9)

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
