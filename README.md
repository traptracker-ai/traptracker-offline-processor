# Trap Tracker Offline Processor

Trap Tracker Offline Processor is a Docker-based web application for processing camera trap image and video datasets locally. It is designed for conservationists, researchers and biodiversity practitioners who need to run AI-assisted wildlife detection without relying on a hosted cloud platform.

This repository contains the application code and Docker configuration. Large model weights are not included.

## Intended use

This tool is free to use for non-commercial conservation, ecological research, biodiversity monitoring, academic research and education.

Commercial use requires prior written permission. See [`COMMERCIAL_USE.md`](COMMERCIAL_USE.md).

## Key features

- Docker-based deployment
- Web-based local dashboard
- Offline image and video processing
- ONNX Runtime GPU inference
- CUDA provider detection
- Support for user-supplied ONNX models
- Configurable class-name files
- Recursive folder processing
- Browser upload for smaller batches
- Server-side input folder scanning for large batches
- Confidence and IoU threshold configuration
- Frame extraction for videos
- Optional annotated outputs
- CSV exports
- Failed-file reports
- Job progress tracking
- Resume support for interrupted runs
- Results table, annotated gallery and downloadable results ZIP

## Model weights

Model weights are not stored in this repository.

Place compatible ONNX models in the local `models/` folder when running the application. The Processing page lets you choose between two pipelines:

- **Single-stage detector** - one model (e.g. YOLOv10, YOLO26) finds and labels animals in a single pass.
- **Two-stage detector** - a detector finds animals, then a species classifier (e.g. DeepFaune) labels each crop.

```text
models/
  single-stage-detectors/
    uk-mammals.onnx
    uk-mammals.txt                    # optional; same name as the model
  two-stage-detectors/
    <name>_stage1_detector.onnx
    <name>_stage2_classifier.onnx
    <name>_stage2_classifier.txt      # optional
```

Single-stage models use a class file with the same name as the model by default; any `.txt` or `.names` file next to the models can also be selected. If no class file is found, the class names embedded in the ONNX model are used. ONNX files placed directly in `models/` are still treated as single-stage models.

Two-stage models are paired by name. The classifier's input size, normalisation and class names are read from its ONNX metadata when present (the `.txt` file is only a fallback). If the stage 1 detector has an `animal` class, only animal boxes are classified and people/vehicles keep the detector label; otherwise every box is reclassified. In the results CSV, `species` and `confidence` come from the classifier and `detector_class` / `detector_confidence` record the stage 1 result.

Official Trap Tracker AI model releases will be published separately through Zenodo so they can be cited using DOIs.

## Quick start

Clone the repository:

```bash
git clone https://github.com/traptracker-ai/traptracker-offline-processor.git
cd traptracker-offline-processor
```

Create the local runtime folders if they do not already exist:

```bash
mkdir models input output runs logs
```

Place your ONNX models in `models/single-stage-detectors/` and/or `models/two-stage-detectors/` (see [Model weights](#model-weights)).

Run the application:

```bash
docker compose up --build
```

Open the local web interface:

```text
http://localhost:8501
```

## Large batches

For large image or video batches, copy files into:

```text
input/
```

Then use the Input Manager in the web interface and start a run from the server-side input folder.

## GPU support

This image uses CUDA 12.4 and cuDNN 9. GPU inference requires:

- an NVIDIA GPU;
- a recent NVIDIA driver on the host;
- Docker with NVIDIA GPU support;
- NVIDIA Container Toolkit support where required by the host operating system.

You can test GPU access with:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If the application falls back to CPU inference, check the run page for warning messages and confirm that Docker can access the GPU.

## Resuming an interrupted run

Detections are streamed to:

```text
runs/<run_id>/detections.csv
```

Each handled file is recorded in:

```text
runs/<run_id>/manifest.jsonl
```

If a run fails part way through, open the run and click **Resume from last file**. Already processed files are skipped and new detections are appended to the same CSV.

## Licence

This software is released under the Trap Tracker AI Research and Non-Commercial Licence. See [`LICENSE`](LICENSE), [`NON_COMMERCIAL_USE.md`](NON_COMMERCIAL_USE.md) and [`COMMERCIAL_USE.md`](COMMERCIAL_USE.md).

The Trap Tracker AI name, Trap Tracker name, logos and branding are not licensed for reuse. See [`TRADEMARKS.md`](TRADEMARKS.md).

## Citation

If you use this software in research, conservation or teaching, please cite it. See [`CITATION.cff`](CITATION.cff).
