#!/usr/bin/env python3
"""Turn a folder of numbered images (e.g. one UA-DETRAC sequence) into an mp4.

UA-DETRAC is distributed as JPEG images, not videos. Run this once per
sequence, then use the mp4 with the other scripts.

Example:
    python src/frames_to_video.py --frames data/DETRAC-Images/MVI_20011 --out data/MVI_20011.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder with the images")
    ap.add_argument("--out", required=True, help="mp4 file to write")
    ap.add_argument("--fps", type=float, default=25, help="UA-DETRAC is recorded at 25 fps")
    ap.add_argument("--pattern", default="*.jpg", help="image file pattern (default *.jpg)")
    return ap.parse_args()


def main():
    args = parse_args()
    images = sorted(Path(args.frames).glob(args.pattern))
    if not images:
        sys.exit(f"No images matching {args.pattern} in {args.frames}")

    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for path in images:
        frame = cv2.imread(str(path))
        if frame is not None:
            writer.write(frame)
    writer.release()
    print(f"Saved: {args.out}  ({len(images)} frames, {w}x{h}, {args.fps:g} fps)")


if __name__ == "__main__":
    main()
