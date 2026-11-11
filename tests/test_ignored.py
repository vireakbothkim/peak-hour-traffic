"""Quick checks for the ignored-region helpers. Run from the repo root:

    python tests/test_ignored.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ignored import in_ignored, load_ignored_regions  # noqa: E402

XML = """<sequence name="MVI_TEST">
  <sequence_attribute camera_state="stable" sence_weather="sunny"/>
  <ignored_region>
    <box left="0" top="0" width="100" height="50"/>
    <box left="200" top="100" width="50" height="50"/>
  </ignored_region>
</sequence>"""


def _regions():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "MVI_TEST.xml"
        path.write_text(XML)
        return load_ignored_regions(path)


def test_regions_are_read_from_the_xml():
    assert _regions() == [(0.0, 0.0, 100.0, 50.0), (200.0, 100.0, 50.0, 50.0)]


def test_points_inside_and_outside_the_regions():
    regions = _regions()
    assert in_ignored(50, 25, regions)        # inside the first box
    assert in_ignored(225, 125, regions)      # inside the second box
    assert not in_ignored(150, 25, regions)   # between the boxes
    assert not in_ignored(50, 80, regions)    # below the first box


def test_no_regions_means_nothing_is_ignored():
    assert not in_ignored(10, 10, [])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
