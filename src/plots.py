#!/usr/bin/env python3
"""Draw the flow chart (vehicles per bin, stacked by class, peak outlined) and the class mix.

Example:
    python src/plots.py --bins outputs/cam1_bins.csv --out outputs/cam1_chart.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to a file, no window needed (works on Colab and servers)
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

NON_CLASS_COLUMNS = ["bin_start", "bin_start_s", "bin_length_s", "total", "A_to_B", "B_to_A"]
COLORS = {"car": "#4a90d9", "motorcycle": "#f5a623", "bus": "#50b383", "truck": "#8e6bbf"}
EXTRA_COLORS = ["#999999", "#d1495b", "#00798c", "#edae49"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", required=True, help="bins CSV from aggregate.py")
    ap.add_argument("--out", required=True, help="PNG file to write")
    ap.add_argument("--title", default="Vehicle flow per time bin")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.bins)
    bin_s = df["bin_length_s"].max()
    class_cols = [c for c in df.columns if c not in NON_CLASS_COLUMNS and df[c].sum() > 0]
    x = list(range(len(df)))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [3, 1]})

    # Left: flow per bin, stacked by class
    bottom = pd.Series([0] * len(df))
    extra = iter(EXTRA_COLORS)
    for c in class_cols:
        ax.bar(x, df[c], bottom=bottom, label=c, color=COLORS.get(c) or next(extra, "#999999"))
        bottom = bottom + df[c]
    if not class_cols:
        ax.bar(x, df["total"], color="#4a90d9", label="total")

    full = df[df["bin_length_s"] >= bin_s - 1e-9]
    if full.empty:
        full = df
    if full["total"].max() > 0:
        pk = full["total"].idxmax()
        ax.bar(pk, df.loc[pk, "total"], fill=False, edgecolor="#c0392b", linewidth=2.2)
        ax.annotate(f"Peak flow: {int(df.loc[pk, 'total'])} at {df.loc[pk, 'bin_start']}",
                    xy=(pk, df.loc[pk, "total"]), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color="#c0392b")
    step = max(1, len(df) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(df["bin_start"][::step], rotation=45, ha="right")
    ax.set_xlabel(f"Time in video, bin start (mm:ss), bins of {bin_s:g} s")
    ax.set_ylabel("Vehicles counted")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # counts are whole vehicles
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(top=max(1, df["total"].max()) * 1.18)
    if class_cols:
        ax.legend(fontsize=8)

    # Right: class mix
    totals = {c: int(df[c].sum()) for c in class_cols}
    if totals:
        names = list(totals)
        shares = [100 * totals[c] / sum(totals.values()) for c in names]
        extra = iter(EXTRA_COLORS)
        bars = ax2.barh(names, shares, color=[COLORS.get(c) or next(extra, "#999999") for c in names])
        for bar, c, s in zip(bars, names, shares):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{s:.0f}% ({totals[c]})",
                     va="center", fontsize=8)
        ax2.set_xlim(0, 135)
        ax2.invert_yaxis()
    ax2.set_title("Vehicle class mix")
    ax2.set_xlabel("Share of vehicles (%)")

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
