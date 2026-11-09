#!/usr/bin/env python3
"""Build ground-truth line crossings from a UA-DETRAC annotation XML.

The annotations give a box and a vehicle ID for every vehicle in every frame. We apply the SAME
LineCounter and the SAME counting line as the system to the center of each annotated box, so the
true counts and the system counts are directly comparable.

Output: <out> in the same format as crossings.csv (time_s, chunk, track_id, class, direction).

UA-DETRAC vehicle types are car, van, bus and others (there is no motorcycle class). They are mapped
to our classes with --class-map (default: van=car, others=truck). Change it if you disagree.

NOTE: written from the usual UA-DETRAC XML layout
  <sequence> <frame num=".."> <target_list> <target id=".."> <box left top width height/> <attribute vehicle_type=".."/>
Check it against a real file. If the layout differs, only parse_annotations() needs to change.

Example:
    python src/gt_from_detrac.py --xml data/annotations/MVI_20011.xml --line "0,0.5,1,0.5" \
        --out ground_truth/MVI_20011_gt_crossings.csv
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from counting import LineCounter

FIELDS = ["time_s", "chunk", "track_id", "class", "direction"]


def parse_class_map(text):
    mapping = {}
    for pair in filter(None, text.split(",")):
        key, _, value = pair.partition("=")
        mapping[key.strip()] = value.strip()
    return mapping


def parse_annotations(xml_path):
    """Yield (frame_number, target_id, center_x, center_y, vehicle_type), ordered by frame."""
    root = ET.parse(xml_path).getroot()
    frames = sorted(root.iter("frame"), key=lambda f: int(f.get("num")))
    for frame in frames:
        num = int(frame.get("num"))
        for target in frame.iter("target"):
            box = target.find("box")
            if box is None:
                continue
            left, top = float(box.get("left")), float(box.get("top"))
            width, height = float(box.get("width")), float(box.get("height"))
            attribute = target.find("attribute")
            vehicle_type = attribute.get("vehicle_type", "others") if attribute is not None else "others"
            yield num, int(target.get("id")), left + width / 2, top + height / 2, vehicle_type


def build_crossings(xml_path, line, width, height, fps, class_map):
    x1, y1, x2, y2 = [float(v) for v in line.split(",")]
    counter = LineCounter((x1 * width, y1 * height), (x2 * width, y2 * height))
    rows = []
    for num, target_id, cx, cy, vehicle_type in parse_annotations(xml_path):
        direction = counter.update(target_id, cx, cy)
        if direction:
            rows.append({"time_s": round((num - 1) / fps, 2), "chunk": 0, "track_id": target_id,
                         "class": class_map.get(vehicle_type, vehicle_type), "direction": direction})
    return rows


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, help="UA-DETRAC annotation XML for one sequence")
    ap.add_argument("--out", required=True, help="ground-truth crossings CSV to write")
    ap.add_argument("--line", default="0,0.5,1,0.5", help="x1,y1,x2,y2 as fractions (use the same as for the system)")
    ap.add_argument("--width", type=int, default=960, help="frame width in pixels (UA-DETRAC: 960)")
    ap.add_argument("--height", type=int, default=540, help="frame height in pixels (UA-DETRAC: 540)")
    ap.add_argument("--fps", type=float, default=25, help="frames per second (UA-DETRAC: 25)")
    ap.add_argument("--class-map", default="van=car,others=truck", help="e.g. van=car,others=truck")
    return ap.parse_args()


def main():
    args = parse_args()
    if not Path(args.xml).exists():
        sys.exit(f"Annotation file not found: {args.xml}")
    rows = build_crossings(args.xml, args.line, args.width, args.height, args.fps, parse_class_map(args.class_map))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {args.out}  ({len(rows)} true crossings, "
          f"{dict(Counter(r['direction'] for r in rows))}, {dict(Counter(r['class'] for r in rows))})")


if __name__ == "__main__":
    main()
