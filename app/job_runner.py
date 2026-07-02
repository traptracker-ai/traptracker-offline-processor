from pathlib import Path
import threading, uuid, time, json, traceback, copy, csv
from collections import Counter
import cv2
from detector import YOLO26ONNX, IMAGE_EXTS, VIDEO_EXTS
from utils import exif_datetime

JOBS = {}
LOCK = threading.Lock()

# Keys that are large or noisy and should not be persisted into status.json.
_HEAVY_KEYS = {'error'}

# Stable CSV column order so appends stay aligned across files and across resumes.
CSV_FIELDS = [
    'filename', 'source_type', 'species', 'class_id', 'confidence',
    'x1', 'y1', 'x2', 'y2',
    'datetime_original', 'video_file', 'frame_number', 'timestamp_seconds',
    'source_path',
]


def _persist(job_id, snapshot):
    run_dir = snapshot.get('run_dir')
    if not run_dir:
        return
    to_write = {k: v for k, v in snapshot.items() if k not in _HEAVY_KEYS}
    cfg = to_write.get('config')
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        files = cfg.get('files')
        if isinstance(files, list):
            cfg['files'] = len(files)
        to_write['config'] = cfg
    tmp = Path(run_dir, 'status.json.tmp')
    final = Path(run_dir, 'status.json')
    try:
        tmp.write_text(json.dumps(to_write, indent=2, default=str), encoding='utf-8')
        tmp.replace(final)
    except Exception:
        pass


def _set(job_id, **kwargs):
    with LOCK:
        JOBS.setdefault(job_id, {}).update(kwargs)
        snapshot = copy.deepcopy(JOBS[job_id])
    _persist(job_id, snapshot)


def get_job(job_id):
    with LOCK:
        return copy.deepcopy(JOBS.get(job_id, {}))


def list_jobs(runs_dir):
    items = []
    for status in sorted(Path(runs_dir).glob('*/status.json'), reverse=True):
        try:
            items.append(json.loads(status.read_text(encoding='utf-8')))
        except Exception:
            pass
    return items


def _load_manifest(run_dir):
    """Return the set of source paths already fully processed in a prior attempt."""
    done = set()
    mpath = Path(run_dir) / 'manifest.jsonl'
    if mpath.exists():
        for line in mpath.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if 'path' in rec:
                    done.add(rec['path'])
            except Exception:
                continue
    return done


def _summary_from_csv(csv_path):
    """Rebuild the species counter from an existing CSV (used when resuming)."""
    counter = Counter()
    detections = 0
    if Path(csv_path).exists():
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    detections += 1
                    counter[row.get('species', '')] += 1
        except Exception:
            pass
    return counter, detections


def start_job(config, resume_job_id=None):
    if resume_job_id:
        job_id = resume_job_id
    else:
        job_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    run_dir = Path(config['runs_dir']) / job_id
    (run_dir / 'annotated').mkdir(parents=True, exist_ok=True)
    _set(job_id, id=job_id, status='queued', progress=0, message='Queued',
         run_dir=str(run_dir), started=time.strftime('%Y-%m-%d %H:%M:%S'),
         config=config, processed=0, detections=0, failed=0,
         total=len(config.get('files', [])), resumed=bool(resume_job_id))
    t = threading.Thread(target=_run, args=(job_id, config), daemon=True)
    t.start()
    return job_id


def _safe_rel(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _process_image(detector, path, relname, run_dir, save_annotated):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError('OpenCV could not read image')
    dets = detector.detect_image_array(img, filename=relname)
    dt = exif_datetime(path)
    for d in dets:
        d['source_type'] = 'image'
        d['datetime_original'] = dt
        d['source_path'] = str(path)
    if save_annotated and dets:
        ann = detector.annotate(img, dets)
        out = Path(run_dir) / 'annotated' / relname
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out.with_suffix('.jpg')), ann)
        if not ok:
            cv2.imwrite(str(out.with_suffix('.png')), ann)
    return dets


def _process_video(detector, path, relname, run_dir, save_annotated, frame_stride, max_annotated=2000):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('OpenCV could not open video')
    rows = []
    frame_i = 0
    annotated_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    stem = Path(relname).stem
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_i % frame_stride != 0:
                frame_i += 1
                continue
            ts = frame_i / fps if fps else 0
            name = f"{stem}_frame_{frame_i:06d}.jpg"
            dets = detector.detect_image_array(frame, filename=name)
            for d in dets:
                d['source_type'] = 'video'
                d['video_file'] = relname
                d['frame_number'] = frame_i
                d['timestamp_seconds'] = round(ts, 3)
                d['source_path'] = str(path)
            if save_annotated and dets and annotated_count < max_annotated:
                ann = detector.annotate(frame, dets)
                out = Path(run_dir) / 'annotated' / stem / name
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), ann)
                annotated_count += 1
            rows.extend(dets)
            frame_i += 1
    finally:
        cap.release()
    return rows


def _run(job_id, config):
    run_dir = Path(config['runs_dir']) / job_id
    csv_path = run_dir / 'detections.csv'
    fail_path = run_dir / 'failed_files.csv'
    manifest_path = run_dir / 'manifest.jsonl'

    try:
        _set(job_id, status='loading', message='Loading ONNX model')
        providers = [p.strip() for p in config.get('providers', 'CUDAExecutionProvider,CPUExecutionProvider').split(',') if p.strip()]
        detector = YOLO26ONNX(
            config['model_path'], config['classes_path'], providers=providers,
            input_size=config.get('input_size', 640), conf=config.get('conf', 0.25),
            iou=config.get('iou', 0.45),
        )

        all_files = [Path(p) for p in config['files']]
        total = len(all_files)

        # Resume support: skip files already recorded in the manifest, and pick up
        # the running species counter / detection count from the existing CSV.
        done_paths = _load_manifest(run_dir)
        if done_paths:
            counter, detections = _summary_from_csv(csv_path)
        else:
            counter, detections = Counter(), 0
        processed = len(done_paths)
        failures_count = 0

        gpu_note = '' if detector.on_gpu else ' (running on CPU — see warnings)'
        resume_note = f' (resuming — {processed} already done)' if done_paths else ''
        _set(job_id, status='running',
             message=f'Processing {total} files{gpu_note}{resume_note}',
             providers=detector.providers, on_gpu=detector.on_gpu,
             warnings=getattr(detector, 'warnings', []), total=total,
             processed=processed, detections=detections, failed=failures_count)

        root = Path(config.get('input_root') or '/')
        frame_stride = int(config.get('frame_stride', 1))
        save_annotated = config.get('save_annotated', True)

        csv_exists = csv_path.exists()
        # Open CSV in append mode so a resume continues the same file.
        csv_f = open(csv_path, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if not csv_exists or csv_path.stat().st_size == 0:
            writer.writeheader()
            csv_f.flush()
        manifest_f = open(manifest_path, 'a', encoding='utf-8')

        last_ui = 0.0
        try:
            for path in all_files:
                spath = str(path)
                if spath in done_paths:
                    continue
                file_dets = []
                error = None
                try:
                    relname = _safe_rel(path, root)
                    ext = path.suffix.lower()
                    if ext in IMAGE_EXTS:
                        file_dets = _process_image(detector, path, relname, run_dir, save_annotated)
                    elif ext in VIDEO_EXTS:
                        file_dets = _process_video(detector, path, relname, run_dir, save_annotated, frame_stride)
                except Exception as e:
                    error = str(e)
                    failures_count += 1

                # Persist this file's detections immediately.
                if file_dets:
                    for d in file_dets:
                        writer.writerow(d)
                        counter[d.get('species', '')] += 1
                    detections += len(file_dets)
                    csv_f.flush()

                # Record the file as handled (done or failed) in the resume ledger.
                manifest_f.write(json.dumps({'path': spath, 'ok': error is None,
                                             'error': error, 'n': len(file_dets)}) + '\n')
                manifest_f.flush()
                processed += 1

                # Throttle UI/status writes so per-file work isn't dominated by I/O
                # at very large scale: at most ~3 updates/sec, plus the final file.
                now = time.time()
                if now - last_ui > 0.3 or processed == total:
                    last_ui = now
                    summary = [{'species': s, 'count': c} for s, c in counter.most_common(50)]
                    _set(job_id, processed=processed, detections=detections,
                         failed=failures_count,
                         progress=int(processed / total * 100) if total else 100,
                         message=f'Processed {processed}/{total}', summary=summary)
        finally:
            csv_f.close()
            manifest_f.close()

        # Finalization phase — make it visible. The CSV is already written, so this
        # is just the failed-files report and a status flip. ZIP is built lazily on
        # download, not here, so completion is near-instant even for huge runs.
        _set(job_id, status='finalizing', progress=100,
             message='Finalizing results (writing reports)')

        # Rewrite the failed-files CSV from the manifest (authoritative record).
        try:
            with open(fail_path, 'w', newline='', encoding='utf-8') as ff:
                fw = csv.writer(ff)
                fw.writerow(['file', 'error'])
                if manifest_path.exists():
                    for line in manifest_path.read_text(encoding='utf-8').splitlines():
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if not rec.get('ok', True):
                            fw.writerow([rec.get('path', ''), rec.get('error', '')])
        except Exception:
            pass

        summary = [{'species': s, 'count': c} for s, c in counter.most_common(50)]
        _set(job_id, status='complete', progress=100, message='Complete',
             detections=detections, failed=failures_count,
             csv=str(csv_path), failed_csv=str(fail_path),
             summary=summary, completed=time.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        try:
            (run_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        except Exception:
            pass
        _set(job_id, status='error', message=str(e), error=traceback.format_exc(),
             completed=time.strftime('%Y-%m-%d %H:%M:%S'))
