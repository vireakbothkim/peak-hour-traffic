#!/usr/bin/env python3
"""Make a short annotated demo video: boxes, track IDs, the counting line and live counts.

Use it for the demo and as a backup recording in case a live run fails.

Example:
    python src/make_demo.py --video data/cam1.mp4 --start-s 30 --seconds 30 --out outputs/demo.mp4
"""
import argparse
import sys

import cv2
from ultralytics import YOLO

from counting import LineCounter

VEHICLE_CLASSES = [2, 5, 7]  # car, bus, truck: same as detect_track.py (motorcycles are not counted)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="outputs/demo.mp4")
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--line", default="0,0.5,1,0.5", help="x1,y1,x2,y2 as fractions of width/height")
    ap.add_argument("--start-s", type=float, default=0, help="where in the video to start")
    ap.add_argument("--seconds", type=float, default=30, help="length of the demo clip")
    ap.add_argument("--stride", type=int, default=1, help="1 = every frame (smoothest)")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--tracker", default="bytetrack.yaml", help="same default as detect_track.py")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start = int(args.start_s * fps)
    end = start + int(args.seconds * fps)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    x1, y1, x2, y2 = [float(v) for v in args.line.split(",")]
    p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
    counter = LineCounter(p1, p2)
    totals = {"A_to_B": 0, "B_to_A": 0}

    model = YOLO(args.model)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w, h))

    for frame_idx in range(start, end):
        if not cap.grab():
            break
        if (frame_idx - start) % args.stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        result = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=args.conf,
                             tracker=args.tracker, device=args.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for tid, (cx, cy, _w, _h) in zip(boxes.id.int().tolist(), boxes.xywh.tolist()):
                direction = counter.update(tid, cx, cy)
                if direction:
                    totals[direction] += 1

        annotated = result.plot()  # frame with boxes and IDs drawn
        cv2.line(annotated, p1, p2, (0, 0, 255), 3)
        text = f"A_to_B: {totals['A_to_B']}   B_to_A: {totals['B_to_A']}   total: {sum(totals.values())}"
        cv2.rectangle(annotated, (10, 10), (10 + 14 * len(text), 50), (0, 0, 0), -1)
        cv2.putText(annotated, text, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Saved: {args.out}  counts: {totals}")


if __name__ == "__main__":
    main()
