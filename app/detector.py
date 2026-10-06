from pathlib import Path
import ast
import json
import cv2
import numpy as np
import onnxruntime as ort

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}

GPU_PROVIDERS = ('CUDAExecutionProvider', 'TensorrtExecutionProvider')
DEFAULT_INPUT_SIZE = 640


def load_classes(path: str):
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    return [x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def available_providers():
    try:
        return ort.get_available_providers()
    except Exception:
        return ['CPUExecutionProvider']


def _create_session(model_path, providers=None):
    """Create an ORT session, recording (rather than hiding) any fall back to CPU.

    Returns (session, active_providers, on_gpu, warnings).
    """
    model_path = str(model_path)
    if not Path(model_path).exists():
        raise FileNotFoundError(f'ONNX model not found: {model_path}')
    providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
    avail = available_providers()
    requested_gpu = any(p in GPU_PROVIDERS for p in providers)
    usable = [p for p in providers if p in avail]
    if not usable:
        usable = ['CPUExecutionProvider']

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    warnings = []
    try:
        session = ort.InferenceSession(model_path, sess_options=so, providers=usable)
    except Exception as e:
        # Record why GPU init failed instead of silently dropping to CPU.
        warnings.append(f'Failed to init providers {usable} for {Path(model_path).name}: {e}')
        session = ort.InferenceSession(model_path, sess_options=so, providers=['CPUExecutionProvider'])

    active = session.get_providers()
    # Detect the genuinely active EP. ORT lists CPU as a fallback even on a
    # working GPU session, so "on GPU" means a GPU EP is actually present.
    on_gpu = any(p in GPU_PROVIDERS for p in active)
    if requested_gpu and not on_gpu:
        warnings.append(
            'GPU was requested but ONNX Runtime is running on CPU. '
            'This usually means the onnxruntime-gpu build does not match the '
            'container CUDA/cuDNN version, or the NVIDIA runtime is not available to Docker.'
        )
    return session, active, on_gpu, warnings


def _metadata(session):
    try:
        return dict(session.get_modelmeta().custom_metadata_map or {})
    except Exception:
        return {}


def _parse_literal(value):
    """Parse a metadata value written as JSON or as a Python literal (Ultralytics)."""
    if not value:
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(value)
        except Exception:
            continue
    return None


def _nms(boxes, scores, iou_thresh=0.45):
    """Pure-numpy NMS. boxes are x1,y1,x2,y2 in model space."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = order[1:][iou <= iou_thresh]
    return keep


def annotate(image_bgr, detections):
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, [d['x1'], d['y1'], d['x2'], d['y2']])
        label = f"{d['species']} {d['confidence']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 110, 60), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, max(0, y1-th-8)), (x1+tw+8, y1), (20, 110, 60), -1)
        cv2.putText(out, label, (x1+4, max(12, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


class YOLOONNX:
    """Single-stage YOLO detector (YOLOv8/v10/11/26 ONNX exports).

    Also used as stage 1 of the two-stage pipeline.
    """

    def __init__(self, model_path: str, classes_path: str = '', providers=None, input_size=None, conf=0.25, iou=0.45):
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.session, self.providers, self.on_gpu, self.warnings = _create_session(self.model_path, providers)
        meta = _metadata(self.session)

        # Class names: explicit file first, then the names embedded by Ultralytics.
        self.classes = load_classes(classes_path)
        if not self.classes:
            names = _parse_literal(meta.get('names'))
            if isinstance(names, dict):
                self.classes = [str(names[k]) for k in sorted(names, key=int)]
            elif isinstance(names, list):
                self.classes = [str(n) for n in names]

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Input size: a static input shape wins; otherwise the user's value; otherwise
        # the training size recorded in the model metadata; otherwise 640.
        size = int(input_size or 0)
        if size <= 0:
            imgsz = _parse_literal(meta.get('imgsz'))
            if isinstance(imgsz, (list, tuple)) and imgsz:
                size = int(max(imgsz))
            elif isinstance(imgsz, int):
                size = imgsz
        if size <= 0:
            size = DEFAULT_INPUT_SIZE
        size = int(np.ceil(size / 32) * 32)
        shape = inp.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                size = int(h)
        self.input_size = size
        # True only if the model's batch axis is dynamic (a symbolic name, not
        # a fixed int) -- a model exported with a fixed batch=1 input shape
        # will raise a shape-mismatch error if actually sent a batch, so
        # detect_batch() below falls back to one-at-a-time calls for those.
        self.batch_dynamic = isinstance(shape, (list, tuple)) and len(shape) == 4 and not isinstance(shape[0], int)

    def preprocess(self, image_bgr):
        h, w = image_bgr.shape[:2]
        size = self.input_size
        scale = min(size / w, size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dw, dh = (size - nw) // 2, (size - nh) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        return np.ascontiguousarray(tensor), scale, dw, dh, w, h

    def _parse_outputs(self, outputs):
        out = outputs[0]
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 3:
            out = out[0]
        return out

    def _decode(self, raw):
        """Return list of (x1,y1,x2,y2,score,cls_id) in MODEL (letterboxed) space.

        Handles three common YOLO ONNX export layouts:
          1. End-to-end / NMS export:  (N, 6)  -> x1,y1,x2,y2,score,cls
          2. Raw transposed:           (4+nc, num) -> needs transpose + argmax
          3. Raw non-transposed:       (num, 4+nc) -> argmax over class cols
        """
        if raw.size == 0:
            return []
        # Case 1: already (N, 6) end-to-end results.
        if raw.ndim == 2 and raw.shape[1] == 6 and raw.shape[0] != 6:
            results = []
            for det in raw:
                x1, y1, x2, y2, score, cls_id = [float(v) for v in det[:6]]
                if score < self.conf:
                    continue
                results.append((x1, y1, x2, y2, score, int(round(cls_id))))
            return results

        # Cases 2 & 3: raw grid output. Orient so rows are predictions, cols are attrs.
        a, b = raw.shape
        # Heuristic: the predictions axis is the larger one for typical YOLO grids.
        if a < b:
            preds = raw.T  # (num, 4+nc)
        else:
            preds = raw
        if preds.shape[1] < 5:
            return []
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]
        cls_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), cls_ids]
        mask = scores >= self.conf
        if not np.any(mask):
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        cx, cy, ww, hh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - ww / 2
        y1 = cy - hh / 2
        x2 = cx + ww / 2
        y2 = cy + hh / 2
        xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(xyxy, scores, self.iou)
        return [(float(xyxy[i, 0]), float(xyxy[i, 1]), float(xyxy[i, 2]), float(xyxy[i, 3]),
                 float(scores[i]), int(cls_ids[i])) for i in keep]

    def _rows_from_decoded(self, decoded, scale, dw, dh, orig_w, orig_h, filename):
        rows = []
        for x1, y1, x2, y2, score, cls_i in decoded:
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale
            x2 = (x2 - dw) / scale
            y2 = (y2 - dh) / scale
            x1 = max(0, min(orig_w, x1)); x2 = max(0, min(orig_w, x2))
            y1 = max(0, min(orig_h, y1)); y2 = max(0, min(orig_h, y2))
            label = self.classes[cls_i] if 0 <= cls_i < len(self.classes) else str(cls_i)
            rows.append({
                'filename': filename,
                'class_id': cls_i,
                'species': label,
                'confidence': round(float(score), 6),
                'x1': round(float(x1), 2),
                'y1': round(float(y1), 2),
                'x2': round(float(x2), 2),
                'y2': round(float(y2), 2),
            })
        return rows

    def detect_image_array(self, image_bgr, filename='image'):
        tensor, scale, dw, dh, orig_w, orig_h = self.preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = self._parse_outputs(outputs)
        decoded = self._decode(raw)
        return self._rows_from_decoded(decoded, scale, dw, dh, orig_w, orig_h, filename)

    def detect_batch(self, images_bgr, filenames=None):
        """Detect on a list of images with ONE inference call per batch
        instead of one call per image -- at batch=1, most of the wall-clock
        cost of a GPU inference call is fixed kernel-launch/data-transfer
        overhead rather than compute, so processing images one at a time
        leaves the GPU mostly idle between calls; batching amortises that
        fixed cost across many images per call.

        Falls back to one-at-a-time calls transparently if this model's ONNX
        graph has a fixed (non-dynamic) batch axis and genuinely cannot
        accept more than one image per call.
        """
        if not images_bgr:
            return []
        filenames = filenames or [f'image_{i}' for i in range(len(images_bgr))]
        if not self.batch_dynamic:
            return [self.detect_image_array(img, fn) for img, fn in zip(images_bgr, filenames)]

        tensors, metas = [], []
        for img in images_bgr:
            t, scale, dw, dh, w, h = self.preprocess(img)
            tensors.append(t[0])
            metas.append((scale, dw, dh, w, h))
        batch = np.ascontiguousarray(np.stack(tensors).astype(np.float32))
        try:
            outputs = self.session.run(None, {self.input_name: batch})
        except Exception:
            # Some exports declare a dynamic batch axis but only actually
            # tolerate batch=1 (e.g. a fixed-shape NMS op baked in elsewhere
            # in the graph). Don't fail the whole batch -- fall back.
            return [self.detect_image_array(img, fn) for img, fn in zip(images_bgr, filenames)]

        raw_batch = np.asarray(outputs[0], dtype=np.float32)
        if raw_batch.ndim != 3 or raw_batch.shape[0] != len(images_bgr):
            # Unexpected output shape for a batched call -- fall back rather
            # than risk silently mis-attributing detections to the wrong image.
            return [self.detect_image_array(img, fn) for img, fn in zip(images_bgr, filenames)]

        results = []
        for i, (scale, dw, dh, orig_w, orig_h) in enumerate(metas):
            decoded = self._decode(raw_batch[i])
            results.append(self._rows_from_decoded(decoded, scale, dw, dh, orig_w, orig_h, filenames[i]))
        return results

    def annotate(self, image_bgr, detections):
        return annotate(image_bgr, detections)


# Backwards-compatible name used by earlier versions of the app.
YOLO26ONNX = YOLOONNX


class CropClassifierONNX:
    """Stage 2 species classifier (DeepFaune-style) that labels detector crops.

    Input size, normalisation and class names are read from the ONNX metadata
    when present, so a mismatched .txt file cannot silently shift the labels.
    """

    def __init__(self, model_path: str, classes_path: str = '', providers=None, batch_size=16):
        self.model_path = str(model_path)
        self.batch_size = max(1, int(batch_size))
        self.session, self.providers, self.on_gpu, self.warnings = _create_session(self.model_path, providers)
        meta = _metadata(self.session)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        out = self.session.get_outputs()[0]
        n_out = out.shape[-1] if isinstance(out.shape[-1], int) else None

        size = 0
        if isinstance(inp.shape, (list, tuple)) and len(inp.shape) == 4 and isinstance(inp.shape[2], int):
            size = inp.shape[2]
        if size <= 0:
            try:
                size = int(meta.get('input_size', 0))
            except ValueError:
                size = 0
        self.input_size = size if size > 0 else 182

        mean = _parse_literal(meta.get('mean')) or [0.485, 0.456, 0.406]
        std = _parse_literal(meta.get('std')) or [0.229, 0.224, 0.225]
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

        meta_classes = _parse_literal(meta.get('classes'))
        file_classes = load_classes(classes_path)
        if isinstance(meta_classes, list) and meta_classes:
            self.classes = [str(c) for c in meta_classes]
            if file_classes and file_classes != self.classes:
                self.warnings.append(
                    f'Class file {Path(classes_path).name} does not match the class list embedded in '
                    f'{Path(self.model_path).name}; using the embedded list.'
                )
        else:
            self.classes = file_classes
        if n_out and self.classes and len(self.classes) != n_out:
            self.warnings.append(
                f'Classifier outputs {n_out} classes but {len(self.classes)} class names were found; '
                'labels may be wrong.'
            )

        # Optional class_groups metadata: {label: detector class it belongs under}.
        # TwoStageDetector uses it to keep each box's label within its detector class.
        self.class_groups = None
        groups = _parse_literal(meta.get('class_groups'))
        if groups is not None:
            if isinstance(groups, dict) and self.classes and (not n_out or len(self.classes) == n_out):
                self.class_groups = {str(k): str(v).lower() for k, v in groups.items()}
                ungrouped = [c for c in self.classes if c not in self.class_groups]
                if ungrouped:
                    self.warnings.append(
                        f'class_groups in {Path(self.model_path).name} has no group for '
                        f'{", ".join(ungrouped)}; those labels will never be predicted.'
                    )
            else:
                self.warnings.append(
                    f'Ignoring class_groups metadata in {Path(self.model_path).name}: it is malformed '
                    'or the class list does not match the model outputs.'
                )

    def _crop(self, image_bgr, box):
        """Square crop around the box (DeepFaune convention), clipped to the image."""
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        bw, bh = x2 - x1, y2 - y1
        if bw > bh:
            pad = (bw - bh) // 2
            y1, y2 = y1 - pad, y2 + pad
        elif bh > bw:
            pad = (bh - bw) // 2
            x1, x2 = x1 - pad, x2 + pad
        crop = image_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return crop if crop.size else None

    def _preprocess(self, crop_bgr):
        size = self.input_size
        resized = cv2.resize(crop_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        return np.transpose(rgb, (2, 0, 1))

    def classify(self, image_bgr, boxes, masks=None):
        """Return one (class_id, label, prob) per box, or None where no crop was possible.

        masks, if given, holds one boolean array per box marking the labels that
        box may take. The prob returned is still from the softmax over all
        labels, so a crop that doesn't look like any allowed label keeps a low
        confidence instead of being inflated by the restriction.
        """
        results = [None] * len(boxes)
        tensors, idx = [], []
        for i, box in enumerate(boxes):
            crop = self._crop(image_bgr, box)
            if crop is None:
                continue
            tensors.append(self._preprocess(crop))
            idx.append(i)
        for start in range(0, len(tensors), self.batch_size):
            batch = np.ascontiguousarray(np.stack(tensors[start:start + self.batch_size]).astype(np.float32))
            logits = np.asarray(self.session.run(None, {self.input_name: batch})[0], dtype=np.float32)
            logits = logits - logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=1, keepdims=True)
            for row, i in zip(probs, idx[start:start + self.batch_size]):
                if masks is not None and masks[i] is not None:
                    c = int(np.argmax(np.where(masks[i], row, -1.0)))
                else:
                    c = int(np.argmax(row))
                label = self.classes[c] if 0 <= c < len(self.classes) else str(c)
                results[i] = (c, label, float(row[c]))
        return results


class TwoStageDetector:
    """Stage 1 finds boxes, stage 2 classifies each crop to species.

    If the classifier carries class_groups metadata ({label: detector class}),
    each box is classified only among the labels grouped under its detector
    class, so an 'animal' box can never come back as e.g. 'Car'. Boxes whose
    detector class has no labels keep their detector label.

    Without class_groups: if the stage 1 detector has an 'animal' class
    (MegaDetector-style), only animal boxes are sent to the classifier and
    person/vehicle/etc. keep their detector label; if it has no 'animal' class
    (e.g. a species detector), every box is reclassified over all labels.
    """

    def __init__(self, detector_path, classifier_path, classifier_classes_path='', providers=None,
                 input_size=None, conf=0.25, iou=0.45):
        self.detector = YOLOONNX(detector_path, '', providers=providers,
                                 input_size=input_size, conf=conf, iou=iou)
        self.classifier = CropClassifierONNX(classifier_path, classifier_classes_path, providers=providers)
        det_classes = [c.lower() for c in self.detector.classes]
        self.input_size = self.detector.input_size
        self.providers = self.detector.providers
        self.on_gpu = self.detector.on_gpu and self.classifier.on_gpu
        self.warnings = self.detector.warnings + self.classifier.warnings

        # Per detector class id: boolean mask of the labels its boxes may take.
        self.group_masks = None
        groups = self.classifier.class_groups
        if groups:
            label_groups = [groups.get(c) for c in self.classifier.classes]
            masks = {i: np.array([g == name for g in label_groups])
                     for i, name in enumerate(det_classes)}
            self.group_masks = {i: m for i, m in masks.items() if m.any()}
            if not self.group_masks:
                self.warnings.append(
                    'Classifier class_groups match none of the detector classes '
                    f'({", ".join(self.detector.classes)}); detector labels are kept unchanged.'
                )
            self.classify_ids = set(self.group_masks)
        else:
            self.classify_ids = {i for i, c in enumerate(det_classes) if c == 'animal'} or None

    def _classify_rows(self, image_bgr, rows):
        for d in rows:
            d['detector_class'] = d['species']
            d['detector_confidence'] = d['confidence']
        targets = [d for d in rows if self.classify_ids is None or d['class_id'] in self.classify_ids]
        if targets:
            boxes = [(d['x1'], d['y1'], d['x2'], d['y2']) for d in targets]
            masks = ([self.group_masks[d['class_id']] for d in targets]
                     if self.group_masks is not None else None)
            for d, res in zip(targets, self.classifier.classify(image_bgr, boxes, masks)):
                if res is None:
                    continue
                c, label, prob = res
                d['class_id'] = c
                d['species'] = label
                d['confidence'] = round(prob, 6)
        return rows

    def detect_image_array(self, image_bgr, filename='image'):
        rows = self.detector.detect_image_array(image_bgr, filename=filename)
        return self._classify_rows(image_bgr, rows)

    def detect_batch(self, images_bgr, filenames=None):
        """Batch stage 1 (localisation) across images in one inference call;
        stage 2 (classification) still runs per image, batched across that
        image's own boxes as before -- already the cheaper stage per call,
        and per-image detection counts are too small/uneven to usefully
        combine into one cross-image classifier batch."""
        if not images_bgr:
            return []
        filenames = filenames or [f'image_{i}' for i in range(len(images_bgr))]
        per_image_rows = self.detector.detect_batch(images_bgr, filenames=filenames)
        return [self._classify_rows(img, rows) for img, rows in zip(images_bgr, per_image_rows)]

    def annotate(self, image_bgr, detections):
        return annotate(image_bgr, detections)


def build_detector(config, providers=None):
    """Create the single-stage or two-stage pipeline described by a run config."""
    if config.get('pipeline') == 'two_stage':
        return TwoStageDetector(
            config['model_path'], config['classifier_path'], config.get('classifier_classes_path', ''),
            providers=providers, input_size=config.get('input_size'),
            conf=config.get('conf', 0.25), iou=config.get('iou', 0.45),
        )
    return YOLOONNX(
        config['model_path'], config.get('classes_path', ''), providers=providers,
        input_size=config.get('input_size'), conf=config.get('conf', 0.25),
        iou=config.get('iou', 0.45),
    )
