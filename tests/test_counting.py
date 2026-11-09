"""Quick checks for the counting logic. Run from the repo root:

    python tests/test_counting.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from counting import LineCounter  # noqa: E402


def run(counter, track_id, ys, x=50):
    """Feed a vehicle moving through the given y positions; return what was reported."""
    return [counter.update(track_id, x, y) for y in ys]


def test_crossing_down_is_a_to_b():
    c = LineCounter((0, 150), (100, 150))  # horizontal line, left to right
    assert run(c, 1, [100, 130, 160, 190]) == [None, None, "A_to_B", None]


def test_crossing_up_is_b_to_a():
    c = LineCounter((0, 150), (100, 150))
    assert run(c, 1, [200, 170, 140, 110]) == [None, None, "B_to_A", None]


def test_vehicle_is_counted_only_once_even_if_it_jitters():
    c = LineCounter((0, 150), (100, 150))
    results = run(c, 1, [140, 160, 140, 160, 140])
    assert results.count("A_to_B") == 1 and results.count("B_to_A") == 0


def test_vehicle_that_never_crosses_is_not_counted():
    c = LineCounter((0, 150), (100, 150))
    assert run(c, 1, [100, 110, 120, 130]) == [None] * 4


def test_crossing_beyond_the_line_segment_is_ignored():
    c = LineCounter((0, 150), (100, 150))  # segment only covers x from 0 to 100
    assert run(c, 1, [140, 160], x=500) == [None, None]


def test_two_vehicles_are_tracked_separately():
    c = LineCounter((0, 150), (100, 150))
    assert c.update(1, 50, 140) is None
    assert c.update(2, 50, 160) is None
    assert c.update(1, 50, 160) == "A_to_B"
    assert c.update(2, 50, 140) == "B_to_A"


def test_first_sighting_below_the_line_is_not_a_crossing():
    c = LineCounter((0, 150), (100, 150))
    assert c.update(1, 50, 400) is None  # appears already on side B


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
