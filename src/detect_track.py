#!/usr/bin/env python3
"""Detect, track and count vehicles crossing a virtual line.

Input : a video file (or an image sequence such as frames/img%05d.jpg)
Output: <out-dir>/<video-name>_crossings.csv, one row per counted vehicle:
            time_s, chunk, track_id, class, direction
        <out-dir>/<video-name>_meta.json (video length, fps, size)

The video is processed in chunks (default 10 minutes). Every finished chunk is
saved to <out-dir>/<video-name>/chunks/. If Colab disconnects, run the SAME
command again and it continues from the first unfinished chunk.

Example:
    python src/detect_track.py --video data/cam1.mp4 --out-dir outputs --stride 3
"""
import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from counting import LineCounter

# COCO class ids: 2 car, 3 motorcycle, 5 bus, 7 truck
VEHICLE_CLASSES = [2, 3, 5, 7]
FIELDS = ["time_s", "chunk", "track_id", "class", "direction"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video file, or image pattern like frames/img%%05d.jpg")
    ap.add_argument("--out-dir", default="outputs", help="where results are written")
    ap.add_argument("--model", default="yolov8s.pt", help="YOLOv8 weights (downloaded automatically)")
    ap.add_argument("--line", default="0,0.5,1,0.5",
                    help="counting line as x1,y1,x2,y2 in fractions of width/height (default: horizontal, middle)")
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame (default 3)")
    ap.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--chunk-minutes", type=float, default=10, help="size of each resumable chunk")
    ap.add_argument("--limit-seconds", type=float, default=0, help="only process the first N seconds (0 = all)")
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU, cpu, mps (default: automatic)")
    ap.add_argument("--force", action="store_true", help="delete earlier chunks and start over")
    return ap.parse_args()


def line_to_pixels(text, width, height):
    x1, y1, x2, y2 = [float(v) for v in text.split(",")]
    return (x1 * width, y1 * height), (x2 * width, y2 * height)


def write_csv(path, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)  # atomic: a chunk file is either complete or absent


def process_chunk(args, video, chunk_idx, start, end, fps, line_px):
    model = YOLO(args.model)  # new model object = fresh tracker, so chunks are independent
    counter = LineCounter(*line_px)
    cap = cv2.VideoCapture(video)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    rows = []
    for frame_idx in range(start, end):
        if not cap.grab():
            break
        if (frame_idx - start) % args.stride:
            continue  # skip this frame (grab() is cheap, no full decode)
        ok, frame = cap.retrieve()
        if not ok:
            continue

        result = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=args.conf,
                             tracker=args.tracker, device=args.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            continue  # nothing tracked in this frame

        ids = boxes.id.int().tolist()
        classes = boxes.cls.int().tolist()
        for track_id, cls, (cx, cy, _w, _h) in zip(ids, classes, boxes.xywh.tolist()):
            direction = counter.update(track_id, cx, cy)
            if direction:
                rows.append({"time_s": round(frame_idx / fps, 2), "chunk": chunk_idx,
                             "track_id": track_id, "class": model.names[cls], "direction": direction})
    cap.release()
    return rows


def main():
    args = parse_args()
    video = args.video

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if total <= 0:
        sys.exit("Could not read the number of frames. Convert the video to .mp4 first.")
    if args.limit_seconds > 0:
        total = min(total, int(args.limit_seconds * fps))

    name = Path(video).parent.name if "%" in video else Path(video).stem
    out_dir = Path(args.out_dir)
    chunk_dir = out_dir / name / "chunks"
    if args.force and chunk_dir.exists():
        shutil.rmtree(chunk_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to mix chunks made with different settings
    config = {"video": name, "model": args.model, "line": args.line, "stride": args.stride,
              "conf": args.conf, "tracker": args.tracker, "chunk_minutes": args.chunk_minutes,
              "limit_seconds": args.limit_seconds}
    config_path = chunk_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            sys.exit("Settings differ from the earlier run in this folder.\n"
                     "Use the same settings to resume, or add --force to start over.")
    else:
        config_path.write_text(json.dumps(config, indent=2))

    line_px = line_to_pixels(args.line, width, height)
    chunk_frames = max(1, int(args.chunk_minutes * 60 * fps))
    n_chunks = math.ceil(total / chunk_frames)
    print(f"{name}: {total} frames, {fps:.1f} fps, {width}x{height}, {n_chunks} chunk(s), stride {args.stride}")

    for i in range(n_chunks):
        chunk_path = chunk_dir / f"chunk_{i:04d}.csv"
        if chunk_path.exists():
            print(f"chunk {i + 1}/{n_chunks}: already done, skipping")
            continue
        start, end = i * chunk_frames, min((i + 1) * chunk_frames, total)
        t0 = time.time()
        rows = process_chunk(args, video, i, start, end, fps, line_px)
        write_csv(chunk_path, rows)
        print(f"chunk {i + 1}/{n_chunks}: {len(rows)} vehicles counted in {time.time() - t0:.0f}s")

    # Merge all chunks into one file
    all_rows = []
    for path in sorted(chunk_dir.glob("chunk_*.csv")):
        with open(path, newline="") as f:
            all_rows.extend(csv.DictReader(f))
    final_path = out_dir / f"{name}_crossings.csv"
    write_csv(final_path, all_rows)
    (out_dir / f"{name}_meta.json").write_text(json.dumps(
        {"duration_s": round(total / fps, 2), "fps": fps, "width": width, "height": height}, indent=2))

    by_direction = Counter(r["direction"] for r in all_rows)
    print(f"Done. {len(all_rows)} vehicles counted {dict(by_direction)}")
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
