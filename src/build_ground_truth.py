#!/usr/bin/env python3
"""Build ground-truth crossings for every UA-DETRAC clip listed in the clips file.

For each clip whose `source` is ua_detrac, this reads <annotations-dir>/<clip>.xml and writes
<ground-truth-dir>/<clip>_gt_crossings.csv, using the counting line from the SAME row of clips.csv,
so the true counts and the system counts use exactly the same line.

Run it again after any line changes (it overwrites the old files).

Example:
    python src/build_ground_truth.py --clips configs/clips.csv --annotations-dir annotations --ground-truth-dir ground_truth
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from gt_from_detrac import FIELDS, build_crossings, parse_class_map

DEFAULT_LINE = "0,0.5,1,0.5"


def build_all(clips_path, annotations_dir, gt_dir, source="ua_detrac", only=(), width=960, height=540, fps=25,
              class_map="van=car,others=truck"):
    """Returns a list of (clip, status, n_crossings). Missing annotation files are reported, not fatal."""
    clips = pd.read_csv(clips_path, comment="#", skip_blank_lines=True)
    if "source" in clips.columns:
        clips = clips[clips["source"].astype(str) == source]
    if only:
        clips = clips[clips["clip"].astype(str).isin(set(only))]
    results = []
    Path(gt_dir).mkdir(parents=True, exist_ok=True)
    for _, row in clips.iterrows():
        clip = str(row["clip"])
        xml_path = Path(annotations_dir) / f"{clip}.xml"
        if not xml_path.exists():
            results.append((clip, f"annotation file not found: {xml_path}", 0))
            continue
        line = row["line"] if "line" in clips.columns and isinstance(row["line"], str) and row["line"].strip() else DEFAULT_LINE
        rows = build_crossings(xml_path, line, width, height, fps, parse_class_map(class_map))
        out = Path(gt_dir) / f"{clip}_gt_crossings.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        results.append((clip, "ok", len(rows)))
    return results


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="configs/clips.csv")
    ap.add_argument("--annotations-dir", default="annotations", help="folder with <clip>.xml files")
    ap.add_argument("--ground-truth-dir", default="ground_truth")
    ap.add_argument("--source", default="ua_detrac", help="only clips with this source value (default ua_detrac)")
    ap.add_argument("--only", default="", help="comma-separated clip names (default: all)")
    ap.add_argument("--class-map", default="van=car,others=truck")
    return ap.parse_args()


def main():
    args = parse_args()
    only = [c.strip() for c in args.only.split(",") if c.strip()]
    results = build_all(args.clips, args.annotations_dir, args.ground_truth_dir, args.source, only,
                        class_map=args.class_map)
    if not results:
        sys.exit(f"No clips with source '{args.source}' found in {args.clips}.")
    for clip, status, n in results:
        print(f"{clip}: {status}" + (f" ({n} true crossings)" if status == "ok" else ""))
    counts = Counter(status == "ok" for _, status, _ in results)
    print(f"\n{counts[True]} ground-truth file(s) written to {args.ground_truth_dir}, {counts[False]} problem(s)")


if __name__ == "__main__":
    main()
