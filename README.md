# Vehicle Counting and Short-Term Peak-Flow Detection (AIN 320, Group 4)

Count vehicles in traffic-camera video with a pretrained YOLOv8 model and a tracker, find the busiest
moments inside each video (peak flow), break the counts down by vehicle type, and measure how accurate the
counts are against ground truth. Public datasets only (AI City Challenge Track 1, UA-DETRAC).

```
video -> detect_track.py -> crossings.csv -> aggregate.py -> bins.csv + summary.json -> plots.py -> chart
                                  |
   ground truth (gt_from_detrac.py) -----> evaluate.py -> error tables, peak-flow agreement
```

No training is needed: the model is pretrained on COCO and already knows car, bus and truck. (COCO also knows motorcycles, but they are left out: our datasets do not label them.)
Limitation to state in the report: the datasets do not contain Cambodian vehicle types such as remorques, rickshaws or motorbikes with carts.

## Repo layout

```
src/
  counting.py          line-crossing logic (the "virtual line"), tested
  ignored.py           skip vehicles inside UA-DETRAC's ignored regions (used by detect_track.py via --ignore-xml)
  detect_track.py      video in -> crossings.csv out (chunked, resumable, automatic frame skipping)
  aggregate.py         crossings.csv -> bins.csv (+ summary.json): flow per bin, per direction, per class, peak flow
  plots.py             bins.csv -> chart: flow per bin stacked by class, peak outlined, class mix
  gt_from_detrac.py    UA-DETRAC annotation XML -> true crossings for one clip, using the SAME line as the system
  build_ground_truth.py   the same for every ua_detrac clip in configs/clips.csv, using each clip's own line
  evaluate.py          system vs ground truth -> errors by clip and by condition, peak-flow agreement
  run_batch.py         run detect + aggregate + plots for every clip in configs/clips.csv
  preview_line.py      draw the counting line on one frame to check it
  frames_to_video.py   turn a UA-DETRAC image folder into an mp4
  make_demo.py         annotated demo clip (boxes, IDs, line, live counts)
configs/
  clips.csv            one row per clip: counting line + conditions (lighting, weather, density)
notebooks/
  run_on_colab.ipynb   runs everything on Google Colab
tests/
  run_all.py           python tests/run_all.py   (27 tests)
```

## File formats (agree on these, then work in parallel)

**crossings.csv** (from `detect_track.py`; ground truth uses the same format): one row per counted vehicle

| column | meaning |
|---|---|
| `time_s` | seconds from the start of the video when the vehicle crossed the line |
| `chunk` | which 10-minute chunk it was processed in (0 for ground truth) |
| `track_id` | the tracker's ID for the vehicle (unique inside a chunk) |
| `class` | car, bus or truck (motorcycles are not counted) |
| `direction` | `A_to_B` or `B_to_A` |

**bins.csv** (from `aggregate.py`): `bin_start` (mm:ss), `bin_start_s`, `bin_length_s`, `A_to_B`, `B_to_A`, `car`, `motorcycle`, `bus`, `truck`, `total`.
**bins_summary.json:** total vehicles, mean per bin, peak flow (count and bin), peak-to-average ratio, top 3 bins, direction totals, class totals and shares.
Peak flow and the average use full bins only, so a short last bin cannot distort them.

**configs/clips.csv:** `clip, video, line, lighting, weather, density, source, notes`
- `clip` = the video file name without .mp4 (output files are named after it)
- `video` = path of the .mp4 inside the videos folder, e.g. `ua_detrac/MVI_20011.mp4`
- `line` = counting line `x1,y1,x2,y2` as fractions of the frame (empty = `0,0.5,1,0.5`)
- `lighting` day/night, `weather` sunny/cloudy/rainy, `density` low/medium/high (these are used to group the errors)

Directions: for the default line (left to right, middle of the frame), side A is above the line and side B below it,
so `A_to_B` means moving down the image. `preview_line.py` shows A and B on a real frame.

## Typical workflow

1. **Get data.** Put videos in the Drive folder (`videos/ua_detrac/`, `videos/ai_city/`) and annotations in `annotations/`.
   UA-DETRAC comes as JPEG images, so convert each sequence first:
   `python src/frames_to_video.py --frames data/DETRAC-Images/MVI_20011 --out data/MVI_20011.mp4`
2. **Choose the counting line** for each clip with `preview_line.py` and write it in `configs/clips.csv` with the conditions.
3. **Run all clips:**
   `python src/run_batch.py --clips configs/clips.csv --videos-dir <videos folder> --out-dir outputs --annotations-dir <annotations folder> --stride 1`
   You get, per clip, `_crossings.csv`, `_bins.csv`, `_bins_summary.json` and `_chart.png`, plus `all_clips_summary.csv`.
4. **Build ground truth** (UA-DETRAC) with the same lines as in clips.csv, for all clips at once:
   `python src/build_ground_truth.py --clips configs/clips.csv --annotations-dir annotations --ground-truth-dir ground_truth`
   (for one clip: `python src/gt_from_detrac.py --xml annotations/MVI_20011.xml --line "0,0.55,1,0.55" --out ground_truth/MVI_20011_gt_crossings.csv`)
5. **Evaluate:**
   `python src/evaluate.py --clips configs/clips.csv --outputs-dir outputs --ground-truth-dir ground_truth --bin-seconds 10`

On Colab, `notebooks/run_on_colab.ipynb` does all of this. Run it on one short clip first.

## Evaluation definitions

- absolute error = |system - true|, percentage error = |system - true| / true x 100
- bias % = (sum of system - sum of true) / sum of true x 100 (negative = under-counting)
- `eval_by_group.csv` groups the errors by lighting, weather, density and source, with the share of clips within 10% error
- `eval_peak_flow.csv`: correlation between the system's and the true flow per bin, whether the busiest bin matches, and the peak gap in vehicles (0 = the system's busiest bin is as busy as the true peak;
  short clips often have near-tied bins, so an exact bin match is strict).
  Use a bin size that gives at least 3 full bins per clip (for 1-minute clips use 10 or 15 seconds)

## Important notes and limits

- **Check the line is the same everywhere.** Ground truth must be built with the same line as the system run.
- **UA-DETRAC classes:** car, van, bus, others (no motorcycle). `gt_from_detrac.py` maps van to car and others to truck by default
  (`--class-map` changes it). Class-level errors for UA-DETRAC are therefore approximate; say so in the report.
- **AI City Track 1 ground truth** is counted by movement (paths through an intersection), not by a single line. Look at the real
  files before using them: either place the line to match one movement, or add zone-based movement counting.
  `evaluate.py` expects ground truth in the crossings.csv format, so AI City files need a small converter once we have them.
- `gt_from_detrac.py` assumes the usual UA-DETRAC XML layout (`frame`, `target`, `box`, `attribute vehicle_type`). Check it against a real file.
- Each vehicle is counted once, the first time it crosses the line. Tracker IDs restart at each 10-minute chunk, so a vehicle crossing
  exactly at a chunk boundary can be missed or counted twice (clips shorter than 10 minutes are a single chunk).
- Night and bad weather lower detection accuracy. Report it honestly.

## Settings worth knowing

| Setting | Default | Notes |
|---|---|---|
| `--stride` | auto | aims for about 8 processed frames per second: every 3rd frame at 25 fps, every frame at 10 fps. **Use `--stride 1` for our short UA-DETRAC clips**: on one clip, every 3rd frame counted 54 vehicles vs 76 true, every frame counted 82 |
| `--line` | `0,0.5,1,0.5` | put it where vehicles cross it cleanly, away from the edges |
| `--model` | `yolov8s.pt` | `yolov8n.pt` is faster and less accurate, `yolov8m.pt` slower and more accurate |
| `--conf` | 0.3 | lower finds more vehicles but adds false detections (detect_track.py and run_batch.py) |
| `--tracker` | bytetrack.yaml | `botsort.yaml` is the alternative (detect_track.py and run_batch.py) |
| `--bin-seconds` | 60 | 5 to 15 for 1-minute clips |
| `--ignore-xml` | none | UA-DETRAC annotation XML: vehicles inside its ignored regions are not counted (`run_batch.py` finds it with `--annotations-dir`) |

Changing the model, line, stride, confidence or tracker after a partial run needs `--force` (it starts over),
because chunks made with different settings must not be mixed. To compare settings without losing earlier results,
use a different `--out-dir` for each (for example `outputs/yolov8n`).

## Team workflow

- Work on a branch, open a pull request, and have a teammate read it before merging to `main`.
- Never commit videos or results (`.gitignore` blocks them). Share them through the Drive folder.
- Before changing a CSV column, tell the team: several scripts depend on the columns.
