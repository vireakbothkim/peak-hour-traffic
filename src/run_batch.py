#!/usr/bin/env python3
"""Run the whole pipeline on every clip listed in a clips file.

For each clip: detect_track.py -> aggregate.py -> plots.py. A clip that fails is reported at the
end and the batch carries on. Every step can be repeated safely: finished chunks are skipped.
At the end, one table with every clip's peak flow and class counts is written to all_clips_summary.csv.

clips file (CSV, lines starting with # are ignored):
  clip,video,line,lighting,weather,density,source,notes
  video = path of the .mp4 inside --videos-dir
  line  = x1,y1,x2,y2 as fractions of the frame (leave empty for the default 0,0.5,1,0.5)

Example:
    python src/run_batch.py --clips configs/clips.csv --videos-dir /content/drive/MyDrive/traffic/videos \
        --out-dir /content/drive/MyDrive/traffic/outputs
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
DEFAULT_LINE = "0,0.5,1,0.5"


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode == 0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="configs/clips.csv")
    ap.add_argument("--videos-dir", required=True, help="folder that contains the videos named in the clips file")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--only", default="", help="comma-separated clip names to run (default: all)")
    ap.add_argument("--bin-seconds", type=float, default=60)
    ap.add_argument("--stride", default="auto")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--force", action="store_true", help="delete earlier chunks of each clip and start over")
    ap.add_argument("--limit-seconds", type=float, default=0, help="only process the first N seconds of each clip")
    ap.add_argument("--annotations-dir", default=None,
                    help="folder with UA-DETRAC annotation XMLs named like the videos; used to skip their ignored regions")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    clips = pd.read_csv(args.clips, comment="#", skip_blank_lines=True)
    for col in ("clip", "video"):
        if col not in clips.columns:
            sys.exit(f"The clips file needs a '{col}' column.")
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        clips = clips[clips["clip"].astype(str).isin(wanted)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_clips, failed = [], []
    for _, row in clips.iterrows():
        clip = str(row["clip"])
        video = Path(args.videos_dir) / str(row["video"])
        line = row["line"] if "line" in clips.columns and isinstance(row["line"], str) and row["line"].strip() else DEFAULT_LINE
        print(f"\n===== {clip} =====", flush=True)
        if not video.exists():
            print(f"Video not found: {video}")
            failed.append(clip)
            continue

        detect = [sys.executable, SRC / "detect_track.py", "--video", video, "--out-dir", out_dir, "--line", line,
                  "--stride", args.stride, "--model", args.model, "--conf", args.conf,
                  "--tracker", args.tracker, "--limit-seconds", args.limit_seconds]
        if args.device is not None:
            detect += ["--device", args.device]
        if args.force:
            detect += ["--force"]
        if args.annotations_dir:
            xml = Path(args.annotations_dir) / f"{video.stem}.xml"
            if xml.exists():
                detect += ["--ignore-xml", xml]
        stem = video.stem
        crossings, bins = out_dir / f"{stem}_crossings.csv", out_dir / f"{stem}_bins.csv"
        steps_ok = (
            run(detect)
            and run([sys.executable, SRC / "aggregate.py", "--crossings", crossings, "--bin-seconds", args.bin_seconds, "--out", bins])
            and run([sys.executable, SRC / "plots.py", "--bins", bins, "--out", out_dir / f"{stem}_chart.png", "--title", f"{clip}: vehicle flow"])
        )
        (ok_clips if steps_ok else failed).append(clip)

    # One table across all clips that worked
    rows = []
    for _, row in clips.iterrows():
        clip = str(row["clip"])
        summary_path = out_dir / f"{Path(str(row['video'])).stem}_bins_summary.json"
        if clip in ok_clips and summary_path.exists():
            s = json.loads(summary_path.read_text())
            extra = {k: row[k] for k in ("lighting", "weather", "density", "source") if k in clips.columns}
            rows.append({"clip": clip, **extra, "total_vehicles": s["total_vehicles"],
                         "mean_per_bin": s["mean_per_full_bin"], "peak_count": s["peak_count"],
                         "peak_bin_start": s["peak_bin_start"], "peak_to_average": s["peak_to_average"],
                         **{f"n_{k}": v for k, v in s["class_totals"].items()}})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "all_clips_summary.csv", index=False)
        print(f"\nSaved: {out_dir / 'all_clips_summary.csv'}")
    print(f"\nFinished: {len(ok_clips)} clip(s) ok, {len(failed)} failed" + (f": {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
