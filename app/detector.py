from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}


def load_classes(path: str):
    p = Path(path)
    if not p.exists():
        return []
    return [x.strip() for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def available_providers():
    try:
        return ort.get_available_providers()
    except Exception:
        return ['CPUExecutionProvider']


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


class YOLO26ONNX:
    def __init__(self, model_path: str, classes_path: str, providers=None, input_size=640, conf=0.25, iou=0.45):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f'ONNX model not found: {self.model_path}')
        self.classes = load_classes(classes_path)
        self.input_size = int(input_size)
        self.conf = float(conf)
        self.iou = float(iou)
        providers = providers or ['CUDAExecutionProvider', 'CPUExecutionProvider']
        avail = available_providers()
        requested_gpu = any(p in ('CUDAExecutionProvider', 'TensorrtExecutionProvider') for p in providers)
        usable = [p for p in providers if p in avail]
        if not usable:
            usable = ['CPUExecutionProvider']

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.warnings = []
        self.session = None
        try:
            self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=usable)
        except Exception as e:
            # Record why GPU init failed instead of silently dropping to CPU.
            self.warnings.append(f'Failed to init providers {usable}: {e}')
            self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=['CPUExecutionProvider'])

        self.providers = self.session.get_providers()
        # Detect the genuinely active EP. ORT lists CPU as a fallback even on a
        # working GPU session, so "on GPU" means a GPU EP is actually present.
        self.on_gpu = any(p in ('CUDAExecutionProvider', 'TensorrtExecutionProvider') for p in self.providers)
        if requested_gpu and not self.on_gpu:
            self.warnings.append(
                'GPU was requested but ONNX Runtime is running on CPU. '
                'This usually means the onnxruntime-gpu build does not match the '
                'container CUDA/cuDNN version, or the NVIDIA runtime is not available to Docker.'
            )
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Infer the model's expected spatial size from a static input shape when present.
        shape = inp.shape
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                self.input_size = int(h)

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

    def detect_image_array(self, image_bgr, filename='image'):
        tensor, scale, dw, dh, orig_w, orig_h = self.preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: tensor})
        raw = self._parse_outputs(outputs)
        decoded = self._decode(raw)
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

    def annotate(self, image_bgr, detections):
        out = image_bgr.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, [d['x1'], d['y1'], d['x2'], d['y2']])
            label = f"{d['species']} {d['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (20, 110, 60), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(out, (x1, max(0, y1-th-8)), (x1+tw+8, y1), (20, 110, 60), -1)
            cv2.putText(out, label, (x1+4, max(12, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out
