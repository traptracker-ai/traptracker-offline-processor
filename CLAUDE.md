# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Flask web app (served by waitress on port 8501) that runs ONNX wildlife-detection models over camera-trap images/videos, fully offline, inside an NVIDIA CUDA Docker container. Model weights are never committed (`*.onnx` and everything under `models/`, `input/`, `output/`, `runs/`, `logs/` is gitignored).

## Commands

```bash
docker compose up --build          # primary way to run; UI at http://localhost:8501
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi   # check Docker GPU access
```

Running without Docker (CPU fallback unless a matching CUDA/cuDNN is installed):

```bash
pip install -r requirements.txt
cd app && python app.py            # dev server; or `python wsgi.py` for waitress
```

The working directory must be `app/`: modules import each other flatly (`from detector import ...`, `from utils import ...`). Paths default to `<repo>/models`, `input`, `runs`, etc. and are overridden by the `MODEL_DIR`, `INPUT_DIR`, `OUTPUT_DIR`, `RUNS_DIR` env vars (Docker sets them to `/app/...`). Other env: `ORT_PROVIDERS`, `MAX_CONTENT_LENGTH_MB`, `SECRET_KEY`, `HOST`/`PORT`/`WAITRESS_THREADS`.

There is no test suite, linter config or build step.

## Architecture

Four Python modules in `app/`:

- **`app.py`**: Flask routes and Jinja templates (`templates/`, plain JS/CSS in `static/`). The `/processing` POST builds a run **config dict** and passes it to `start_job`. Model selections are accepted only if they appear in the server-side listings from `utils`, which also stops paths escaping `MODEL_DIR`. All run-file access goes through `_safe_run_path`, which rejects traversal out of `RUNS_DIR/<job_id>`.
- **`job_runner.py`**: runs each job in a daemon thread. Live state is held in the in-memory `JOBS` dict under `LOCK`. Every `_set()` also writes `runs/<id>/status.json` atomically, so runs survive restarts and `app._load_status` falls back to disk. `_slim_config` swaps `config['files']` for a count in `JOBS` and `status.json`, because that list can hold tens of thousands of paths. The full list reaches only the worker thread, and resume rebuilds it by rescanning the original `input_root`.
- **`detector.py`**: ONNX Runtime inference. `build_detector(config)` returns either `YOLOONNX` (single stage) or `TwoStageDetector` (a `YOLOONNX` detector followed by a DeepFaune-style `CropClassifierONNX` that classifies square crops). Both expose `detect_image_array`, `detect_batch` and `annotate`, plus `on_gpu`, `providers` and `warnings`. `_create_session` deliberately records a CPU fallback as a warning that the UI shows, instead of hiding it.
- **`utils.py`**: walks the input folder, lists models and pairs two-stage models, reads EXIF, builds ZIPs.

### Key behaviours to preserve

- **Run artefacts** live in `runs/<job_id>/`: `detections.csv` (streamed, append mode), `manifest.jsonl` (one line per handled file, `{path, ok, error}`), `status.json`, `failed_files.csv` (rebuilt from the manifest when the run finishes), `annotated/`, `error.txt`, and `matched_files.csv` when copying is on. With `copy_matches`, `_copy_matches` in `job_runner._run` copies each original file whose detections include a chosen species (case-insensitive, at least `copy_min_conf`, and for videos at least `copy_min_frames` distinct frames) to `OUTPUT_DIR/<copy_folder>/<species>/<relpath>` as soon as that file is processed. Videos only, unless `copy_images` is set. On resume, files that already exist with the same size are skipped. `results.zip` is built lazily on first download and cached.
- **Stop**: `request_stop` sets the job's event in `STOP_EVENTS`; the worker checks it before each file and each video frame, drops any unfinished video or pending image batch (they're not in the manifest), and ends with status `stopped`, which can be resumed. Only a worker alive in this process can be stopped.
- **Annotated output** (`job_runner.annotate_mode`): config `annotate` is `all`, `matches` (only files with a chosen copy species at `copy_min_conf`; a video's frames are deleted again if it falls short of `copy_min_frames`; needs `copy_matches`, else it means `all`) or `none`. Older configs only have `save_annotated`, which maps to `all`/`none`.
- **Gallery** (`app.gallery`) is built from `detections.csv`. Images map to `annotated/<relname>.jpg|.png`. Video frames are saved as `annotated/<video stem>/<frame file>` and grouped into one card per video, showing its most confident frame; `?video=<video_file>` lists all of that video's frames.
- **Species picker**: `/api/model-species` calls `detector.species_labels`, which loads the selected model on CPU the same way a run does and returns the labels it can output. For two-stage models that is the classifier labels plus detector classes that keep their own label. Results are cached by model file path and mtime.
- **Resume** reuses the same job_id. It skips paths already in the manifest and rebuilds counts from the existing CSV. On resume the CSV keeps its existing header, so add new columns only at the end of `CSV_FIELDS` (`job_runner.py`) and treat older CSVs as possibly lacking them.
- **Batching**: images are batched through `detect_batch` (`batch_size`, default 16). It falls back to one image per call when the model has a fixed batch axis, raises on a batch, or returns an unexpected output shape. Videos run frame by frame with `frame_stride`. For two-stage models, only stage 1 is batched across images.
- **YOLO output decoding** (`YOLOONNX._decode`) accepts three layouts: end-to-end `(N,6)`, raw `(4+nc, num)` and raw `(num, 4+nc)`. NMS is pure numpy. Input size is chosen in this order: a static ONNX input shape, then the user's value, then `imgsz` from model metadata, then 640.
- **Class names**: single-stage uses only the model's own `<model>.txt`/`.names`, else the Ultralytics `names` metadata; there is deliberately no free class-file choice or shared `classes.txt` fallback, and a class list whose length differs from the embedded names (or, for the classifier, from its outputs) raises instead of mislabelling. The two-stage classifier prefers its embedded `classes`/`mean`/`std`/`input_size` metadata over the `.txt`. If the classifier has `class_groups` metadata (label -> detector class), each box is classified only among the labels in its detector class, and boxes whose class has no labels keep the detector label (see `models/README.md`). Without it, if the stage 1 detector has an `animal` class, only those boxes are reclassified. Classifier crops use the model's `crop_padding` metadata (default 0) and an antialiased PIL bilinear resize, matching how DeepFaune-style training crops are made. Two-stage rows keep `detector_class` and `detector_confidence`.
- **Model layout**: single-stage models go in `models/` or `models/single-stage-detectors/`. Two-stage pairs go in `models/two-stage-detectors/<name>_stage1_detector.onnx` + `<name>_stage2_classifier.onnx` (+ optional `.txt`).
- **Input-folder scan** (`utils.start_scan_refresher`): a background thread rescans `INPUT_DIR` every 20s and requests read the cached result, because a full walk can take seconds to minutes on Docker Desktop bind mounts. Actions that change the folder (upload, clear) must call `invalidate_scan_cache()`.
- **CUDA pinning**: `onnxruntime-gpu` in `requirements.txt` must match the Dockerfile's CUDA 12.4 / cuDNN 9 base image. If they don't match, ORT silently falls back to CPU.

## Licensing

Non-commercial licence; the Trap Tracker names and logos are trademarks (see `LICENSE`, `TRADEMARKS.md`). Don't add branding assets or model weights to the repo.
