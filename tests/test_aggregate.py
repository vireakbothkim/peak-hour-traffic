"""Checks for time binning and the peak-flow summary."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from aggregate import make_bins, summarize  # noqa: E402


def crossings(rows):
    return pd.DataFrame(rows, columns=["time_s", "class", "direction"])


def test_counts_land_in_the_right_bins():
    df = crossings([(5, "car", "A_to_B"), (20, "bus", "B_to_A"), (65, "car", "A_to_B"), (70, "truck", "A_to_B")])
    bins = make_bins(df, bin_s=60, duration_s=120)
    assert list(bins["total"]) == [2, 2]
    assert list(bins["A_to_B"]) == [1, 2] and list(bins["B_to_A"]) == [1, 0]
    assert list(bins["car"]) == [1, 1] and list(bins["bus"]) == [1, 0] and list(bins["truck"]) == [0, 1]


def test_empty_bins_are_kept_and_labels_are_mm_ss():
    bins = make_bins(crossings([(5, "car", "A_to_B")]), bin_s=30, duration_s=100)
    assert list(bins["total"]) == [1, 0, 0, 0]
    assert list(bins["bin_start"]) == ["00:00", "00:30", "01:00", "01:30"]


def test_last_bin_can_be_shorter():
    bins = make_bins(crossings([]), bin_s=60, duration_s=100)
    assert list(bins["bin_length_s"]) == [60, 40]


def test_unknown_class_gets_its_own_column():
    bins = make_bins(crossings([(1, "tuk-tuk", "A_to_B")]), bin_s=60, duration_s=60)
    assert "tuk-tuk" in bins.columns and bins.loc[0, "tuk-tuk"] == 1


def test_peak_ignores_a_short_last_bin():
    # The short last bin has the most vehicles, but it must not count as the peak
    df = crossings([(10, "car", "A_to_B"), (11, "car", "A_to_B"), (62, "car", "A_to_B"), (63, "car", "A_to_B"),
                    (64, "car", "A_to_B"), (65, "car", "A_to_B"), (121, "car", "A_to_B")])
    bins = make_bins(df, bin_s=60, duration_s=122)
    s = summarize(bins, 60)
    assert s["peak_bin_start"] == "01:00" and s["peak_count"] == 4
    assert s["n_full_bins"] == 2


def test_summary_class_share_and_peak_to_average():
    df = crossings([(1, "car", "A_to_B")] * 3 + [(61, "bus", "A_to_B")])
    s = summarize(make_bins(df, 60, 120), 60)
    assert s["class_share_pct"]["car"] == 75.0 and s["class_share_pct"]["bus"] == 25.0
    assert s["peak_to_average"] == 1.5  # peak 3, mean 2


def test_header_only_file_gives_zero_counts():
    bins = make_bins(crossings([]), 60, 60)
    assert list(bins["total"]) == [0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
