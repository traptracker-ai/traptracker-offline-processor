# Models folder

Place your ONNX model files and class-name files here when running the application locally.

Example:

```text
models/
  single-stage-detectors/
    model.onnx
    model.txt
  two-stage-detectors/
    <name>_stage1_detector.onnx
    <name>_stage2_classifier.onnx
    <name>_stage2_classifier.txt
```

Large model weights are intentionally not stored in GitHub. Official Trap Tracker AI model releases will be published separately through Zenodo.
