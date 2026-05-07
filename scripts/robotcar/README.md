# RobotCar conversion scripts

Tools for converting Oxford RobotCar raw Bayer PNGs into demosaiced + undistorted RGB PNGs, plus a location-driven workflow for filling in just the frames flagged by a `query_results.csv`.

| Script | Purpose |
| --- | --- |
| [robotcar_convert.py](robotcar_convert.py) | Convert one full traversal into `<seq>/rgb/<ms>.png` + camera_poses + intrinsics. |
| [robotcar_batch.py](robotcar_batch.py) | Multi-traversal batch (debayer-only, no undistort). Different output layout. |
| [build_gps_index.py](build_gps_index.py) | Build the GPS/RTK spatial index and run location queries that produce `query_results.csv`. |
| [robotcar_convert_filtered.py](robotcar_convert_filtered.py) | **Convert only the frames listed in a `query_results.csv`** with smart reuse from existing converted outputs. This README focuses on this script. |

---

## `robotcar_convert_filtered.py` — running with a location CSV

Given one or more `query_results.csv` files (output of `build_gps_index.py --query`), this populates a flat per-location output directory containing only the frames the query matched, plus an updated copy of the CSV with `converted=yes` and `image_path=<dst>` for every frame that landed.

### Output layout

```
<output_root>/
├── <traversal>/
│     ├── <ms>.png             ← flat, no rgb/ subdir, no poses/intrinsics
│     └── ...
├── ...
└── query_results.csv          ← updated copy
```

### Prerequisites

- Python env with `numpy`, `opencv`, `pillow`, and `colour-demosaicing`. The `robotcar` micromamba env on this machine has these:
  ```
  ~/micromamba/envs/robotcar/bin/python
  ```
- Read access to the raw RobotCar downloads tree (`stereo.timestamps` + `stereo/centre/<ts_us>.png`).
- Write access to `<output_root>`.

### Source-resolution priority

For each `(traversal, ms)` row the script checks, in order:

1. **`--existing_location_root`** — if `<existing_location_root>/<traversal>/<ms>.png` exists, it's copied. (Use this when a location's previously-converted frames live somewhere; e.g. point at the original `location_NN/` directory itself.)
2. **`--full_conversion_root`** — if `<full_conversion_root>/<traversal>/rgb/<ms>.png` exists, it's copied. (Use this when somebody else ran `robotcar_convert.py` on the same traversal already.)
3. **Fresh conversion** — debayers + undistorts the raw PNG via `process_image()` from `robotcar_convert.py`.

If the destination PNG already exists, it's left alone (`already_present`). The script is **idempotent** — re-running fills only what's still missing.

### Basic invocation

```bash
~/micromamba/envs/robotcar/bin/python \
  scripts/robotcar/robotcar_convert_filtered.py \
  --queries /path/to/<location>/query_results.csv \
  --downloads_root /robodata/aaadkins/Datasets/semPersFilter/evalCandidates/robotCar/downloads \
  --existing_location_root /robodata/aaadkins/.../robotcar_queries/<location> \
  --full_conversion_root   /robodata/aaadkins/.../extracted_data/robotcar \
  --output_root /robodata/aaadkins/.../robotcar_queries/<location>_filled_in \
  --workers 4
```

Notes:
- `--existing_location_root` and `--full_conversion_root` are both optional. If you skip them, every frame falls through to fresh conversion.
- `--workers 4` parallelizes across distinct traversals. Each worker stays single-threaded (the demosaic itself is CPU-heavy and I/O contention on `/robodata` limits scaling — 4 workers gave ~5 fps total in our runs).
- The `--queries` flag accepts multiple CSVs.

### Recommended workflow for a new location

1. **Sanity-check the CSV first.** Open it and look at the per-traversal row counts (`awk -F',' 'NR>1 && $9=="no" {print $2}' query_results.csv | sort | uniq -c | sort -rn`). One traversal contributing tens of thousands of rows usually means corrupted RTK / GPS data — see [Skipping a problematic traversal](#skipping-a-problematic-traversal).
2. **Validate the pipeline** by regenerating ~10 already-`converted=yes` rows (`--regenerate` forces fresh conversion, skipping priorities 1 and 2) into a throwaway directory and `cmp`-ing against the existing canonical PNGs. Pixel-level diff via `cv2.absdiff` should be zero. Done once per environment / library upgrade.
3. **Dry run** on the full CSV (`--dry_run`) to preview per-traversal source breakdowns and total work.
4. **Real run** with `--workers 4`, logging to a file via `nohup ... > log 2>&1 &`.
5. After the run, inspect the updated `query_results.csv` in `<output_root>` for any rows still marked `converted=no` (these are unrecoverable — usually undownloaded raw frames).

### Skipping a problematic traversal

If you've decided one traversal's frames are unusable (e.g. corrupted RTK so the location matches are spurious), pre-filter the input CSV before passing it in. **Do not** rely on deleting the output subdirectory afterwards — if any of priorities 1 or 2 still have those frames, a later re-run will resurrect them.

```bash
# strip rows for the bad traversal
awk -F',' 'NR==1 || $2!="2015-02-06-13-57-16"' /path/to/query_results.csv \
  > /tmp/query_results_filtered.csv

# point the script at the filtered CSV
python robotcar_convert_filtered.py \
  --queries /tmp/query_results_filtered.csv \
  --output_root .../<location>_filled_in \
  ...
```

### Filling gaps after a re-download

Sometimes parts of a traversal aren't in the downloads tree (the script reports `missing` for those rows) and you'll want to redownload and rerun. Just re-execute the same command — the script skips already-placed frames (`already_present`) and only converts the newly-available ones. The updated CSV is overwritten with the post-run state.

```bash
# after the redownload completes, just rerun:
python robotcar_convert_filtered.py \
  --queries /path/to/query_results.csv \
  --output_root .../<location>_filled_in \
  ...
# the run typically finishes in seconds-to-minutes since most rows are already_present.
```

### Outcome categories (in the per-traversal log line and final summary)

| Category | Meaning |
| --- | --- |
| `already_present` | Destination PNG already existed; nothing to do. |
| `copied_canonical` | Found at `<existing_location_root>/<traversal>/<ms>.png`; copied. |
| `copied_full` | Found at `<full_conversion_root>/<traversal>/rgb/<ms>.png`; copied. |
| `converted` | Freshly demosaiced + undistorted from raw. |
| `missing` | No source available at any priority — usually an undownloaded raw frame. CSV row stays `converted=no`. |

### CLI flags

```
--queries PATH [PATH ...]          One or more query_results.csv inputs.
--downloads_root PATH              Root of raw RobotCar downloads.
--existing_location_root PATH      Priority-1 source (optional).
--full_conversion_root PATH        Priority-2 source (optional).
--output_root PATH                 Destination root.
--regenerate                       Skip priorities 1 and 2; force fresh conversion.
--only_no                          Process only rows with converted==no.
--workers N                        Parallel workers across traversals (default 1).
--dry_run                          Print per-traversal predicted outcomes; write nothing.
```

---

## Tips

- **Performance**: a single worker does ~3 fps; 4 workers ~5 fps total (I/O-bound on `/robodata`). The runtime is dominated by the largest traversal — if one traversal has 10× the rows of others, it'll pin one worker for hours while the rest finish.
- **Idempotency**: safe to interrupt and restart. The destination-exists check is a single `stat()` so re-runs over a fully-populated tree are nearly instant.
- **Backups**: the script overwrites `<output_root>/query_results.csv` on each run. If you want to preserve a prior state (e.g. before re-running with extra source data), copy it aside first.
- **CRLF**: input CSVs from `build_gps_index.py` are CRLF — the script handles this automatically. The output CSV is LF.
