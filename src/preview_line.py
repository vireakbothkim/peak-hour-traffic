#!/usr/bin/env python3
"""Draw the counting line on one frame so you can check it before a long run.

The line is red. A and B mark the two sides: A_to_B means a vehicle moved from
side A to side B. Adjust --line until it crosses the road in the right place.

Example:
    python src/preview_line.py --video data/cam1.mp4 --line "0.1,0.6,0.9,0.6" --out preview.png
"""
import argparse
import math
import sys

import cv2


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--line", default="0,0.5,1,0.5", help="x1,y1,x2,y2 as fractions of width/height")
    ap.add_argument("--time-s", type=float, default=5, help="which second of the video to show")
    ap.add_argument("--out", default="preview.png")
    return ap.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.time_s * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Could not read a frame at that time. Try a smaller --time-s.")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in args.line.split(",")]
    p1, p2 = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))
    cv2.line(frame, p1, p2, (0, 0, 255), 3)

    # Side B is where the cross product is positive: direction (-dy, dx)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    for label, sign in (("A", -1), ("B", 1)):
        pos = (int(mx + sign * 45 * nx) - 10, int(my + sign * 45 * ny) + 10)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imwrite(args.out, frame)
    print(f"Saved: {args.out}  (frame size {w}x{h})")


if __name__ == "__main__":
    main()
