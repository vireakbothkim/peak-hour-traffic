"""Checks for the error calculations."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import error_rows, group_summary, peak_flow_row  # noqa: E402


def crossings(rows):
    return pd.DataFrame(rows, columns=["time_s", "class", "direction"])


def test_errors_for_total_direction_and_class():
    system = crossings([(1, "car", "A_to_B")] * 8 + [(2, "bus", "B_to_A")] * 1)
    truth = crossings([(1, "car", "A_to_B")] * 10 + [(2, "bus", "B_to_A")] * 1)
    rows = pd.DataFrame(error_rows("c1", system, truth))
    total = rows[rows["level"] == "total"].iloc[0]
    assert (total["system"], total["truth"], total["abs_error"], total["pct_error"]) == (9, 11, 2, 18.18)
    car = rows[(rows["level"] == "class") & (rows["item"] == "car")].iloc[0]
    assert car["pct_error"] == 20.0


def test_zero_truth_gives_no_percentage():
    rows = pd.DataFrame(error_rows("c1", crossings([(1, "car", "A_to_B")]), crossings([])))
    assert rows[rows["level"] == "total"].iloc[0]["pct_error"] != rows[rows["level"] == "total"].iloc[0]["pct_error"]  # NaN


def test_group_summary_bias_and_within_10_percent():
    clips = []
    for clip, s, t, light in [("a", 9, 10, "day"), ("b", 5, 10, "night")]:
        system = crossings([(1, "car", "A_to_B")] * s)
        truth = crossings([(1, "car", "A_to_B")] * t)
        for r in error_rows(clip, system, truth):
            clips.append({**r, "lighting": light, "weather": "x", "density": "x", "source": "x"})
    groups = group_summary(pd.DataFrame(clips))
    night = groups[(groups["group"] == "lighting") & (groups["value"] == "night")].iloc[0]
    assert night["mean_pct_error"] == 50.0 and night["bias_pct"] == -50.0 and night["share_within_10pct"] == 0.0
    overall = groups[groups["group"] == "overall"].iloc[0]
    assert overall["bias_pct"] == -30.0 and overall["share_within_10pct"] == 50.0


def test_peak_flow_agreement():
    truth = crossings([(t, "car", "A_to_B") for t in (5, 35, 36, 37, 65, 95)])
    system = crossings([(t, "car", "A_to_B") for t in (5, 35, 36, 66, 96)])
    row = peak_flow_row("c1", system, truth, duration=120, bin_s=30)
    assert row["peak_match"] is True and row["true_peak_bin"] == "00:30"
    assert row["correlation"] > 0.8


def test_short_clip_is_reported_not_crashed():
    row = peak_flow_row("c1", crossings([(1, "car", "A_to_B")]), crossings([(1, "car", "A_to_B")]), 30, 30)
    assert "too few" in row["note"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
