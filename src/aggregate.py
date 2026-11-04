#!/usr/bin/env python3
"""Group vehicle crossings into fixed time bins (e.g. 15 minutes) and find the peak.

Input : <name>_crossings.csv from detect_track.py
Output: a bins CSV with one row per time bin:
            bin_start, A_to_B, B_to_A, total

Example:
    python src/aggregate.py --crossings outputs/cam1_crossings.csv --bin-minutes 15 \
        --start-time "2026-11-10 06:00" --out outputs/cam1_bins.csv
"""
import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DIRECTIONS = ["A_to_B", "B_to_A"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crossings", required=True, help="crossings CSV from detect_track.py")
    ap.add_argument("--out", required=True, help="bins CSV to write")
    ap.add_argument("--bin-minutes", type=float, default=15, help="bin size in minutes (default 15)")
    ap.add_argument("--start-time", default="",
                    help='clock time when the whole recording started, e.g. "2026-11-10 06:00" (without it, bins are labelled with elapsed time)')
    ap.add_argument("--offset-s", type=float, default=0,
                    help="seconds between the start of the recording and the start of THIS video file (0 if it is the first or only file)")
    ap.add_argument("--duration-s", type=float, default=0,
                    help="video length in seconds (default: read from the meta file, else last crossing)")
    return ap.parse_args()


def label_for(bin_start_s, start_time):
    if start_time:
        return (start_time + timedelta(seconds=bin_start_s)).strftime("%Y-%m-%d %H:%M")
    seconds = int(bin_start_s)
    hours, minutes, secs = seconds // 3600, (seconds // 60) % 60, seconds % 60
    if secs:  # sub-minute bins (only useful for short tests)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}"  # elapsed time HH:MM


def main():
    args = parse_args()
    df = pd.read_csv(args.crossings)
    bin_s = args.bin_minutes * 60
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M") if args.start_time else None

    # Video length: argument > meta file next to the crossings file > last crossing
    duration = args.duration_s
    meta_path = Path(args.crossings).with_name(Path(args.crossings).name.replace("_crossings.csv", "_meta.json"))
    if not duration and meta_path.exists():
        duration = json.loads(meta_path.read_text())["duration_s"]
    if not duration:
        duration = float(df["time_s"].max()) if len(df) else 0.0

    df["time_s"] = df["time_s"] + args.offset_s
    df["bin"] = (df["time_s"] // bin_s).astype(int)

    first_bin = int(args.offset_s // bin_s)
    last_bin = max(first_bin + 1, math.ceil((args.offset_s + duration) / bin_s))  # exclusive
    bins = range(first_bin, last_bin)

    counts = df.groupby(["bin", "direction"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=bins, columns=DIRECTIONS, fill_value=0).fillna(0).astype(int)
    counts["total"] = counts[DIRECTIONS].sum(axis=1)
    counts.insert(0, "bin_start", [label_for(b * bin_s, start_time) for b in counts.index])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out, index=False)

    print(counts.to_string(index=False))
    if counts["total"].sum() > 0:
        peak = counts.loc[counts["total"].idxmax()]
        print(f"\nPeak bin: {peak['bin_start']} with {int(peak['total'])} vehicles "
              f"(bin size {args.bin_minutes:g} min)")
    else:
        print("\nNo vehicles were counted.")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
