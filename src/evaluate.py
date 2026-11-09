#!/usr/bin/env python3
"""Compare the system's counts with ground-truth counts.

For every clip in the clips file you need
  <outputs-dir>/<clip>_crossings.csv            from detect_track.py
  <ground-truth-dir>/<clip>_gt_crossings.csv    true crossings, same columns (see gt_from_detrac.py)
Clips without a ground-truth file are skipped.

Outputs (written to --out-dir):
  eval_by_clip.csv     system vs true count with error, for the total, each direction and each class
  eval_by_group.csv    errors grouped by lighting, weather, density and source
  eval_peak_flow.csv   how well the per-bin flow follows the ground truth (correlation, peak match)

Definitions:
  absolute error = |system - true|          percentage error = |system - true| / true x 100
  bias %         = (sum of system - sum of true) / sum of true x 100   (negative = under-counting)

Example:
    python src/evaluate.py --clips configs/clips.csv --outputs-dir outputs --ground-truth-dir ground_truth
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import DIRECTIONS, make_bins

CONDITIONS = ["lighting", "weather", "density", "source"]


def load_clips(path):
    clips = pd.read_csv(path, comment="#", skip_blank_lines=True)
    if "clip" not in clips.columns:
        sys.exit("The clips file needs a 'clip' column.")
    for col in CONDITIONS:
        if col not in clips.columns:
            clips[col] = "unknown"
        clips[col] = clips[col].fillna("unknown")
    return clips


def error_rows(clip, system, truth):
    """Long-format error rows for total, per direction and per class."""
    rows = []

    def add(level, item, s, t):
        err = abs(s - t)
        rows.append({"clip": clip, "level": level, "item": item, "system": s, "truth": t,
                     "abs_error": err, "pct_error": round(100 * err / t, 2) if t > 0 else np.nan})

    add("total", "all", len(system), len(truth))
    for d in DIRECTIONS:
        add("direction", d, int((system["direction"] == d).sum()), int((truth["direction"] == d).sum()))
    for c in sorted(set(system["class"]) | set(truth["class"])):
        add("class", c, int((system["class"] == c).sum()), int((truth["class"] == c).sum()))
    return rows


def group_summary(by_clip):
    totals = by_clip[by_clip["level"] == "total"]
    rows = []

    def add(group, value, part):
        t_sum = part["truth"].sum()
        rows.append({
            "group": group, "value": value, "n_clips": len(part),
            "mean_abs_error": round(part["abs_error"].mean(), 2),
            "mean_pct_error": round(part["pct_error"].mean(), 2),
            "bias_pct": round(100 * (part["system"].sum() - t_sum) / t_sum, 2) if t_sum > 0 else np.nan,
            "share_within_10pct": round(100 * (part["pct_error"] <= 10).mean(), 1),
        })

    add("overall", "all", totals)
    for col in CONDITIONS:
        for value, part in totals.groupby(col):
            add(col, value, part)
    return pd.DataFrame(rows)


def peak_flow_row(clip, system, truth, duration, bin_s):
    sys_bins = make_bins(system, bin_s, duration)
    gt_bins = make_bins(truth, bin_s, duration)
    full = sys_bins["bin_length_s"] >= bin_s - 1e-9
    s, g = sys_bins.loc[full, "total"].reset_index(drop=True), gt_bins.loc[full, "total"].reset_index(drop=True)
    row = {"clip": clip, "n_full_bins": int(full.sum()), "bin_seconds": bin_s}
    if len(s) < 3:
        return {**row, "note": "too few full bins for a peak-flow comparison; use a smaller --bin-seconds"}
    corr = float(np.corrcoef(s, g)[0, 1]) if s.std() > 0 and g.std() > 0 else np.nan
    peak_s, peak_g = int(s.idxmax()), int(g.idxmax())
    return {**row, "correlation": round(corr, 3) if corr == corr else np.nan,
            "mean_abs_error_per_bin": round(float((s - g).abs().mean()), 2),
            "system_peak_bin": sys_bins.loc[peak_s, "bin_start"], "true_peak_bin": gt_bins.loc[peak_g, "bin_start"],
            "peak_match": peak_s == peak_g, "peak_within_1_bin": abs(peak_s - peak_g) <= 1, "note": ""}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="configs/clips.csv", help="clips file (see configs/clips.csv)")
    ap.add_argument("--outputs-dir", default="outputs", help="where the system crossings are")
    ap.add_argument("--ground-truth-dir", default="ground_truth", help="where the true crossings are")
    ap.add_argument("--out-dir", default=None, help="where to write the evaluation files (default: outputs-dir)")
    ap.add_argument("--bin-seconds", type=float, default=30, help="bin size for the peak-flow comparison")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir or args.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = load_clips(args.clips)

    error_all, peak_all, skipped = [], [], []
    for _, c in clips.iterrows():
        clip = str(c["clip"])
        sys_path = Path(args.outputs_dir) / f"{clip}_crossings.csv"
        gt_path = Path(args.ground_truth_dir) / f"{clip}_gt_crossings.csv"
        if not sys_path.exists() or not gt_path.exists():
            skipped.append(clip)
            continue
        system, truth = pd.read_csv(sys_path), pd.read_csv(gt_path)
        conditions = {col: c[col] for col in CONDITIONS}
        for r in error_rows(clip, system, truth):
            error_all.append({**r, **conditions})

        meta_path = Path(args.outputs_dir) / f"{clip}_meta.json"
        times = pd.concat([system["time_s"], truth["time_s"]])
        duration = json.loads(meta_path.read_text())["duration_s"] if meta_path.exists() else float(times.max() if len(times) else 0)
        peak_all.append(peak_flow_row(clip, system, truth, duration, args.bin_seconds))

    if not error_all:
        sys.exit("No clip had both a system file and a ground-truth file. Skipped: " + ", ".join(skipped))

    by_clip = pd.DataFrame(error_all)
    by_group = group_summary(by_clip)
    peak = pd.DataFrame(peak_all)
    by_clip.to_csv(out_dir / "eval_by_clip.csv", index=False)
    by_group.to_csv(out_dir / "eval_by_group.csv", index=False)
    peak.to_csv(out_dir / "eval_peak_flow.csv", index=False)

    print("Counting error by group (total counts):")
    print(by_group.to_string(index=False))
    print("\nPeak-flow agreement:")
    print(peak.to_string(index=False))
    if skipped:
        print("\nSkipped (missing system or ground-truth file): " + ", ".join(skipped))
    print(f"\nSaved to {out_dir}: eval_by_clip.csv, eval_by_group.csv, eval_peak_flow.csv")


if __name__ == "__main__":
    main()
