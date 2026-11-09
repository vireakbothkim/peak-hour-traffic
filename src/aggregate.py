#!/usr/bin/env python3
"""Group vehicle crossings into short time bins and report the peak flow.

Input : <name>_crossings.csv from detect_track.py (time_s = seconds from the start of the video)
Output: a bins CSV (one row per bin) and <out>_summary.json

Columns of the bins CSV:
  bin_start      mm:ss from the start of the video
  bin_start_s    the same, in seconds
  bin_length_s   length of this bin (the last bin can be shorter than the others)
  A_to_B, B_to_A vehicles per direction
  car, motorcycle, bus, truck (+ any other class found)   vehicles per class
  total          all vehicles in the bin

Peak flow is measured on FULL bins only, so a short last bin cannot distort it.

Example:
    python src/aggregate.py --crossings outputs/cam1_crossings.csv --bin-seconds 60 --out outputs/cam1_bins.csv
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

DIRECTIONS = ["A_to_B", "B_to_A"]
STANDARD_CLASSES = ["car", "motorcycle", "bus", "truck"]
NON_CLASS_COLUMNS = ["bin_start", "bin_start_s", "bin_length_s", "total", *DIRECTIONS]


def mmss(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def make_bins(crossings, bin_s, duration_s):
    """crossings: DataFrame with time_s, class, direction. Returns one row per bin."""
    n_bins = max(1, math.ceil(duration_s / bin_s))
    df = crossings.copy()
    df["time_s"] = pd.to_numeric(df["time_s"])
    df["bin"] = (df["time_s"] // bin_s).astype(int).clip(upper=n_bins - 1)

    classes = STANDARD_CLASSES + sorted(c for c in df["class"].unique() if c not in STANDARD_CLASSES)
    index = range(n_bins)
    by_dir = df.groupby(["bin", "direction"]).size().unstack(fill_value=0)
    by_dir = by_dir.reindex(index=index, columns=DIRECTIONS, fill_value=0).fillna(0).astype(int)
    by_cls = df.groupby(["bin", "class"]).size().unstack(fill_value=0)
    by_cls = by_cls.reindex(index=index, columns=classes, fill_value=0).fillna(0).astype(int)

    out = pd.concat([by_dir, by_cls], axis=1)
    out["total"] = by_dir.sum(axis=1)
    starts = [i * bin_s for i in index]
    lengths = [round(min(bin_s, duration_s - s), 2) if duration_s > s else bin_s for s in starts]
    out.insert(0, "bin_length_s", lengths)
    out.insert(0, "bin_start_s", starts)
    out.insert(0, "bin_start", [mmss(s) for s in starts])
    return out.reset_index(drop=True)


def summarize(bins, bin_s):
    """Peak flow and class mix. Peak/average use full bins only (all bins if none is full)."""
    full = bins[bins["bin_length_s"] >= bin_s - 1e-9]
    if full.empty:
        full = bins
    mean = float(full["total"].mean())
    peak = full.loc[full["total"].idxmax()]
    top = full.nlargest(3, "total")
    class_cols = [c for c in bins.columns if c not in NON_CLASS_COLUMNS]
    total = int(bins["total"].sum())
    class_totals = {c: int(bins[c].sum()) for c in class_cols}
    return {
        "bin_seconds": bin_s,
        "n_bins": int(len(bins)),
        "n_full_bins": int(len(full)),
        "total_vehicles": total,
        "mean_per_full_bin": round(mean, 2),
        "peak_bin_start": str(peak["bin_start"]),
        "peak_count": int(peak["total"]),
        "peak_to_average": round(float(peak["total"]) / mean, 2) if mean > 0 else None,
        "top3_bins": [{"bin_start": str(r["bin_start"]), "count": int(r["total"])} for _, r in top.iterrows()],
        "direction_totals": {d: int(bins[d].sum()) for d in DIRECTIONS},
        "class_totals": class_totals,
        "class_share_pct": {c: round(100 * n / total, 1) for c, n in class_totals.items()} if total else {},
    }


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crossings", required=True, help="crossings CSV from detect_track.py")
    ap.add_argument("--out", required=True, help="bins CSV to write")
    ap.add_argument("--bin-seconds", type=float, default=60, help="bin size in seconds (default 60)")
    ap.add_argument("--duration-s", type=float, default=0,
                    help="video length in seconds (default: read from the meta file, else last crossing)")
    return ap.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.crossings)

    duration = args.duration_s
    name = Path(args.crossings).name
    meta_path = Path(args.crossings).with_name(name.replace("_crossings.csv", "_meta.json"))
    if not duration and meta_path.exists():
        duration = json.loads(meta_path.read_text())["duration_s"]
    if not duration:
        duration = float(pd.to_numeric(df["time_s"]).max()) if len(df) else 0.0

    bins = make_bins(df, args.bin_seconds, duration)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bins.to_csv(out, index=False)

    summary = summarize(bins, args.bin_seconds)
    summary_path = out.with_name(out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(bins.drop(columns=["bin_start_s"]).to_string(index=False))
    if summary["total_vehicles"] > 0:
        print(f"\nPeak flow: {summary['peak_count']} vehicles in the bin starting at {summary['peak_bin_start']} "
              f"({summary['bin_seconds']:g} s bins, average {summary['mean_per_full_bin']} per bin, "
              f"peak/average {summary['peak_to_average']})")
        print(f"Class mix (%): {summary['class_share_pct']}")
    else:
        print("\nNo vehicles were counted.")
    print(f"Saved: {out} and {summary_path}")


if __name__ == "__main__":
    main()
