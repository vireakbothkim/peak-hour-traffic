"""Helpers for the 'ignored regions' that UA-DETRAC marks in every annotation XML.

UA-DETRAC does not label vehicles inside these regions (far-away traffic, parked cars). The system must not
count vehicles detected there either, otherwise they show up as false positives when compared with the
ground truth.

Used by detect_track.py through --ignore-xml (run_batch.py passes it automatically with --annotations-dir).
"""
import xml.etree.ElementTree as ET


def load_ignored_regions(xml_path):
    """Return the ignored boxes of one sequence as a list of (left, top, width, height) in pixels."""
    root = ET.parse(xml_path).getroot()
    return [tuple(float(box.get(key)) for key in ("left", "top", "width", "height"))
            for box in root.findall("ignored_region/box")]


def in_ignored(cx, cy, regions):
    """True if the point (cx, cy) lies inside any ignored box."""
    return any(left <= cx <= left + width and top <= cy <= top + height
               for left, top, width, height in regions)
