"""Checks for building ground truth for all clips in a clips file."""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from build_ground_truth import build_all  # noqa: E402

XML = """<sequence name="T"><frame num="1"><target_list>
<target id="1"><box left="100" top="200" width="40" height="40"/><attribute vehicle_type="car"/></target></target_list></frame>
<frame num="2"><target_list>
<target id="1"><box left="100" top="300" width="40" height="40"/><attribute vehicle_type="car"/></target></target_list></frame></sequence>"""


def test_uses_each_clips_own_line_and_reports_missing_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ann").mkdir()
        (d / "ann" / "c1.xml").write_text(XML)
        (d / "ann" / "c2.xml").write_text(XML)
        (d / "clips.csv").write_text(
            "# comment\nclip,video,line,source\n"
            'c1,c1.mp4,"0,0.5,1,0.5",ua_detrac\n'      # line at y=270: the vehicle crosses it
            'c2,c2.mp4,"0,0.9,1,0.9",ua_detrac\n'      # line at y=486: it does not
            "c3,c3.mp4,,ua_detrac\n"                   # no annotation file
            "c4,c4.mp4,,ai_city\n")                    # other source: ignored
        results = build_all(d / "clips.csv", d / "ann", d / "gt")
        status = {clip: (s, n) for clip, s, n in results}
        assert status["c1"] == ("ok", 1) and status["c2"] == ("ok", 0)
        assert status["c3"][0].startswith("annotation file not found") and "c4" not in status
        assert len(pd.read_csv(d / "gt" / "c1_gt_crossings.csv")) == 1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
