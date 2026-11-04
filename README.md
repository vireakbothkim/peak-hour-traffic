# Peak-Hour Traffic Pattern Detection (AIN 320, Group 4)

Count vehicles in street-camera footage with a pretrained YOLOv8 model, group the counts into
time bins (15 min / 1 hour) and find the peak hour.

```
video  ->  detect_track.py  ->  crossings.csv  ->  aggregate.py  ->  bins.csv  ->  plots.py  ->  chart.png
           (YOLOv8 + tracker     one row per         (time bins,       one row per    (peak chart)
            + counting line)     counted vehicle      peak)            time bin
```

No training is needed: the model is pretrained on COCO, which already includes car, motorcycle, bus and truck.

## Repo layout

```
src/
  counting.py         line-crossing logic (the "virtual line"), tested in tests/
  detect_track.py     video in -> crossings.csv out (chunked, resumable)
  aggregate.py        crossings.csv -> bins.csv, prints the peak bin
  plots.py            bins.csv -> peak-hour chart (PNG)
  preview_line.py     draw the counting line on one frame to check it
  frames_to_video.py  turn a UA-DETRAC image folder into an mp4
  make_demo.py        annotated demo clip (boxes, IDs, line, live counts)
notebooks/
  run_on_colab.ipynb  runs the whole pipeline on Google Colab
tests/
  test_counting.py    python tests/test_counting.py
```

## The file contracts (agree on these, then work in parallel)

**crossings.csv** (from `detect_track.py`): one row per counted vehicle

| column | meaning |
|---|---|
| `time_s` | seconds from the start of the video when the vehicle crossed the line |
| `chunk` | which 10-minute chunk it was processed in |
| `track_id` | the tracker's ID for the vehicle (unique inside a chunk) |
| `class` | car, motorcycle, bus or truck |
| `direction` | `A_to_B` or `B_to_A` |

**bins.csv** (from `aggregate.py`): `bin_start, A_to_B, B_to_A, total`

Directions: for the default line (left to right, middle of the frame), side A is above the line and
side B is below it, so `A_to_B` means moving down the image. Run `preview_line.py` to see A and B on a real frame.

## Option 1: Google Colab (recommended)

1. Put the code on GitHub (or copy the repo folder into the shared Drive folder).
2. Put videos in the shared Drive folder, e.g. `MyDrive/traffic/videos/`.
3. Open `notebooks/run_on_colab.ipynb` in Colab, choose **Runtime > Change runtime type > T4 GPU**,
   edit the settings cell and run the cells from top to bottom.
4. Start with `LIMIT_SECONDS = 120` to test on two minutes. Set it to `0` for the full video.

If Colab disconnects during a long video, reconnect, run the same cells again, and it continues
from the first unfinished chunk.

## Option 2: local (VS Code)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # on a gaming laptop with an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org)

python src/preview_line.py --video data/cam1.mp4 --line "0,0.5,1,0.5" --out preview.png
python src/detect_track.py --video data/cam1.mp4 --out-dir outputs --stride 3 --limit-seconds 120
python src/aggregate.py --crossings outputs/cam1_crossings.csv --bin-minutes 1 --out outputs/cam1_bins.csv
python src/plots.py --bins outputs/cam1_bins.csv --out outputs/cam1_chart.png
```

Use `--device 0` for an NVIDIA GPU, `--device mps` on a Mac, `--device cpu` otherwise. On CPU only, test on short clips.

## Using UA-DETRAC

UA-DETRAC comes as JPEG images (about 1 minute per sequence), not videos. Convert one sequence first:

```bash
python src/frames_to_video.py --frames data/DETRAC-Images/MVI_20011 --out data/MVI_20011.mp4
```

Its annotations include ground truth, so you can compare the system's counts with the true counts.
Short clips cannot show an hour-scale peak, so use UA-DETRAC to check accuracy and a long recording for the peak-hour demo.

## Settings worth knowing

| Setting | Default | Notes |
|---|---|---|
| `--stride` | 3 | process every 3rd frame. Too large and the tracker loses vehicles; 3-5 is a good range for 25 fps |
| `--line` | `0,0.5,1,0.5` | `x1,y1,x2,y2` as fractions of width and height. Put it where vehicles cross it cleanly, away from the edges |
| `--model` | `yolov8s.pt` | `yolov8n.pt` is faster and less accurate, `yolov8m.pt` is slower and more accurate |
| `--conf` | 0.3 | lower finds more vehicles but adds false detections |
| `--chunk-minutes` | 10 | size of each resumable chunk |

Changing any of these after a partial run needs `--force` (it starts over), because chunks made with different settings must not be mixed.

## Known limits (mention them in the report)

- Each vehicle is counted once, the first time it crosses the line. Vehicles that stop on the line or are hidden by other vehicles can be missed or split into two IDs.
- Tracker IDs restart at each chunk boundary, so a vehicle crossing exactly at a boundary can be missed or counted twice (a few per chunk at most).
- Night and bad weather lower detection accuracy.
- Always compare automatic counts with a manual count on at least one clip.

## Team workflow

- Work on a branch, open a pull request, and have a teammate read it before merging to `main`.
- Never commit videos or results (`.gitignore` blocks them). Share those through the Drive folder.
- Before changing the CSV columns, tell the team, because three scripts depend on them.
