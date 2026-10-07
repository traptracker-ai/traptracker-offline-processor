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

## Two-stage classifier metadata

The stage 2 classifier ONNX can carry these `metadata_props` (JSON values). All are optional.

| Key | Example | Purpose |
|---|---|---|
| `classes` | `["Car", "MelesMeles", ...]` | Label for each output, in order. Takes precedence over the `.txt` file. |
| `input_size` | `182` | Square crop size. |
| `mean`, `std` | `[0.485, 0.456, 0.406]` | Normalisation. |
| `class_groups` | `{"Car": "vehicle", "MelesMeles": "animal", "Person": "person"}` | The stage 1 detector class each label belongs under. |
| `crop_padding` | `0.2` | Context added around each box before the square crop, as a fraction of the box's longer side on every side. Defaults to `0`. Set it to whatever your training crops used. |

With `class_groups`, each box is classified only among the labels grouped under its detector class (compared case-insensitively). An `animal` box can then never be labelled `Car`, and a `vehicle` box is labelled with the best vehicle label instead of just "vehicle". Boxes from a detector class with no labels keep the detector's label. The reported confidence is still the label's probability over all classes, so a crop that fits none of its allowed labels stays low-confidence.

Crops are made the DeepFaune way: pad the box by `crop_padding`, expand it to a square about its centre, clip it to the image, then resize it to `input_size` with antialiased bilinear resampling (the same as torchvision's `Resize`).

Without `class_groups`, only `animal` boxes are reclassified (over every label) when the detector has an `animal` class; otherwise every box is.

Adding it when exporting with PyTorch:

```python
import json, onnx
m = onnx.load("my_stage2_classifier.onnx")
kv = m.metadata_props.add()
kv.key, kv.value = "class_groups", json.dumps({"Car": "vehicle", "MelesMeles": "animal"})
kv = m.metadata_props.add()
kv.key, kv.value = "crop_padding", "0.2"
onnx.save(m, "my_stage2_classifier.onnx")
```
