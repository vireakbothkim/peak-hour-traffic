"""Checks for building ground truth from a (synthetic) UA-DETRAC-style XML file."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gt_from_detrac import build_crossings, parse_class_map  # noqa: E402

XML = """<sequence name="TEST">
  <frame num="1" density="2"><target_list>
    <target id="1"><box left="100" top="200" width="40" height="40"/><attribute vehicle_type="car"/></target>
    <target id="2"><box left="300" top="400" width="40" height="40"/><attribute vehicle_type="van"/></target>
  </target_list></frame>
  <frame num="2" density="2"><target_list>
    <target id="1"><box left="100" top="300" width="40" height="40"/><attribute vehicle_type="car"/></target>
    <target id="2"><box left="300" top="200" width="40" height="40"/><attribute vehicle_type="van"/></target>
  </target_list></frame>
</sequence>"""


def test_truth_uses_the_same_line_logic():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "t.xml"
        path.write_text(XML)
        # 960x540 frame, horizontal line at y = 0.5 * 540 = 270
        rows = build_crossings(path, "0,0.5,1,0.5", 960, 540, 25, parse_class_map("van=car,others=truck"))
    by_id = {r["track_id"]: r for r in rows}
    assert by_id[1]["direction"] == "A_to_B"      # centre moved from y=220 to y=320: top to bottom
    assert by_id[2]["direction"] == "B_to_A"      # centre moved from y=420 to y=220: bottom to top
    assert by_id[2]["class"] == "car"             # van mapped to car
    assert by_id[1]["time_s"] == 0.04             # frame 2 at 25 fps


def test_class_map_parsing():
    assert parse_class_map("van=car, others=truck") == {"van": "car", "others": "truck"}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
