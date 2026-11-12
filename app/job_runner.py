from pathlib import Path
import threading, uuid, time, json, traceback, copy, csv, re, shutil
from collections import Counter
import cv2
from detector import build_detector, IMAGE_EXTS, VIDEO_EXTS
from utils import exif_datetime

JOBS = {}
LOCK = threading.Lock()

# Keys that are large or noisy and should not be persisted into status.json.
_HEAVY_KEYS = {'error'}

# Stable CSV column order so appends stay aligned across files and across resumes.
CSV_FIELDS = [
    'filename', 'source_type', 'species', 'class_id', 'confidence',
    'detector_class', 'detector_confidence',
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


def list_jobs(runs_dir, limit=None):
    """List runs, newest first. job_id is a timestamp prefix
    ('YYYYmmdd-HHMMSS-...'), so sorting paths lexicographically already
    orders them newest-first with no need to open any file -- read and
    parse only `limit` of them (every page load wants just the 8 most
    recent) instead of every run's status.json that has ever existed."""
    paths = sorted(Path(runs_dir).glob('*/status.json'), reverse=True)
    if limit is not None:
        paths = paths[:limit]
    items = []
    for status in paths:
        try:
            items.append(json.loads(status.read_text(encoding='utf-8')))
        except Exception:
            pass
    return items


MATCH_FIELDS = ['source_path', 'source_type', 'species', 'hits', 'max_confidence', 'copied_to', 'error']


def _folder_name(species):
    """A species label made safe to use as one folder name."""
    name = re.sub(r'[^\w\-. ]+', '_', str(species)).strip(' .')
    return name or 'unlabelled'


def _match_species(dets, wanted, min_conf):
    """Species in `dets` that the user asked to copy for.

    Returns {species: (hits, max_conf)}. `hits` is the number of distinct
    video frames (or 1 for an image) with at least one qualifying box, so a
    single frame with three foxes counts once. An empty `wanted` set means
    any detected label qualifies.
    """
    frames, best = {}, {}
    for d in dets:
        sp = str(d.get('species', ''))
        try:
            conf = float(d.get('confidence') or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < min_conf or not sp:
            continue
        if wanted and sp.lower() not in wanted:
            continue
        frames.setdefault(sp, set()).add(d.get('frame_number', 0))
        best[sp] = max(best.get(sp, 0.0), conf)
    return {sp: (len(f), best[sp]) for sp, f in frames.items()}


def _copy_file(src, dest):
    """Copy src to dest unless an identical-size copy is already there
    (e.g. from before a resume)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == src.stat().st_size:
        return
    tmp = dest.with_name(dest.name + '.part')
    shutil.copy2(src, tmp)
    tmp.replace(dest)


def _load_copied(match_path):
    """Source paths already copied (rows without an error), for resume."""
    copied = set()
    if Path(match_path).exists():
        try:
            with open(match_path, 'r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if not row.get('error'):
                        copied.add(row.get('source_path', ''))
        except Exception:
            pass
    return copied


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


def _slim_config(config):
    """A copy of config with 'files' reduced to a count.

    The full config (files list intact) is what the background thread
    itself runs on, via its own `args=(job_id, config)` reference below --
    this slimmed copy is only for JOBS[job_id], which get_job()/_set() both
    deepcopy on every call (every status update from the running job, every
    1.5s UI poll via /api/jobs/<id>, every /runs/<id> page load). With a
    large input folder that 'files' list can be tens of thousands of path
    strings; deepcopying and then JSON-serializing it repeatedly -- for the
    entire duration of the job, under a lock shared with the job thread's
    own progress updates -- was previously real, continuous, avoidable cost
    that scaled with input size and had nothing to do with actual detection
    work. The API/UI never reads config.files at all.
    """
    slim = dict(config)
    files = slim.get('files')
    if isinstance(files, list):
        slim['files'] = len(files)
    return slim


def start_job(config, resume_job_id=None):
    if resume_job_id:
        job_id = resume_job_id
    else:
        job_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:8]
    run_dir = Path(config['runs_dir']) / job_id
    (run_dir / 'annotated').mkdir(parents=True, exist_ok=True)
    _set(job_id, id=job_id, status='queued', progress=0, message='Queued',
         run_dir=str(run_dir), started=time.strftime('%Y-%m-%d %H:%M:%S'),
         config=_slim_config(config), processed=0, detections=0, failed=0,
         total=len(config.get('files', [])), resumed=bool(resume_job_id))
    t = threading.Thread(target=_run, args=(job_id, config), daemon=True)
    t.start()
    return job_id


def _safe_rel(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _process_image_batch(detector, items, run_dir, save_annotated):
    """Run a batch of (path, relname) images through the detector in ONE
    inference call instead of one call per image (see
    detector.YOLOONNX.detect_batch for why this matters at scale), then do
    the same per-image EXIF/annotate/write work a single-image path would.

    Returns a list of (path, relname, dets, error) in the same order as
    `items`, so the caller can record each one exactly as it would a
    single-image result -- a decode failure for one file does not drop the
    rest of the batch.
    """
    images, ok_items, results = [], [], [None] * len(items)
    for i, (path, relname) in enumerate(items):
        img = cv2.imread(str(path))
        if img is None:
            results[i] = (path, relname, [], 'OpenCV could not read image')
            continue
        images.append(img)
        ok_items.append((i, path, relname))

    if images:
        try:
            batch_dets = detector.detect_batch(images, filenames=[r for _, _, r in ok_items])
        except Exception as e:  # noqa: BLE001 - isolate a batch-level failure per file
            batch_dets = [None] * len(images)
            batch_error = str(e)
        else:
            batch_error = None

        for (i, path, relname), img, dets in zip(ok_items, images, batch_dets):
            if dets is None:
                results[i] = (path, relname, [], batch_error)
                continue
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
            results[i] = (path, relname, dets, None)

    return results


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
    match_path = run_dir / 'matched_files.csv'

    try:
        two_stage = config.get('pipeline') == 'two_stage'
        _set(job_id, status='loading',
             message='Loading detector and classifier models' if two_stage else 'Loading ONNX model')
        providers = [p.strip() for p in config.get('providers', 'CUDAExecutionProvider,CPUExecutionProvider').split(',') if p.strip()]
        detector = build_detector(config, providers=providers)

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
        # Images are processed in batches through the detector (one inference
        # call per batch instead of one per image -- see detector.detect_batch);
        # videos are still handled frame-by-frame, unchanged. 16 is a
        # conservative default that fits comfortably in GPU memory for the
        # model sizes this app ships; raise it (config['batch_size']) if you
        # have memory to spare, or lower it if a run hits an out-of-memory error.
        batch_size = max(1, int(config.get('batch_size', 16)))

        csv_exists = csv_path.exists() and csv_path.stat().st_size > 0
        # On resume, keep the existing header so rows stay aligned with CSVs
        # written by earlier versions that had fewer columns.
        fieldnames = CSV_FIELDS
        if csv_exists:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                header = next(csv.reader(f), None)
            if header:
                fieldnames = header
        # Open CSV in append mode so a resume continues the same file.
        csv_f = open(csv_path, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames, extrasaction='ignore')
        if not csv_exists:
            writer.writeheader()
            csv_f.flush()
        manifest_f = open(manifest_path, 'a', encoding='utf-8')

        # Optional: copy original videos (and optionally images) that contain
        # the chosen species to <copy_dir>/<species>/<relative path>, so a large
        # survey can be filtered down to the clips of interest as it runs.
        copy_dir = Path(config['copy_dir']) if config.get('copy_matches') and config.get('copy_dir') else None
        copy_species = {s.lower() for s in config.get('copy_species') or []}
        copy_min_conf = float(config.get('copy_min_conf', 0.5))
        copy_min_frames = max(1, int(config.get('copy_min_frames', 1)))
        copy_images = bool(config.get('copy_images'))
        copied_paths = _load_copied(match_path) if copy_dir else set()
        match_f = match_writer = None
        if copy_dir:
            copy_dir.mkdir(parents=True, exist_ok=True)
            match_exists = match_path.exists() and match_path.stat().st_size > 0
            match_f = open(match_path, 'a', newline='', encoding='utf-8')
            match_writer = csv.DictWriter(match_f, fieldnames=MATCH_FIELDS)
            if not match_exists:
                match_writer.writeheader()
            _set(job_id, copied=len(copied_paths))

        def _copy_matches(path, relname, file_dets):
            is_video = path.suffix.lower() in VIDEO_EXTS
            if not copy_dir or not file_dets or not (is_video or copy_images):
                return
            needed = copy_min_frames if is_video else 1
            any_ok = False
            for sp, (hits, best) in sorted(_match_species(file_dets, copy_species, copy_min_conf).items()):
                if hits < needed:
                    continue
                dest = copy_dir / _folder_name(sp) / relname
                err = ''
                try:
                    _copy_file(path, dest)
                    any_ok = True
                except Exception as e:  # noqa: BLE001 - a failed copy must not fail the run
                    err = str(e)
                match_writer.writerow({'source_path': str(path),
                                       'source_type': 'video' if is_video else 'image',
                                       'species': sp, 'hits': hits,
                                       'max_confidence': round(best, 4),
                                       'copied_to': str(dest), 'error': err})
            if any_ok:
                copied_paths.add(str(path))
            match_f.flush()

        last_ui = 0.0

        def _record(path, relname, file_dets, error):
            """Common bookkeeping for one finished file, whether it came out
            of an image batch or the per-frame video path: write its
            detections, update the manifest, and throttle UI/status writes.
            Does NOT flush -- the caller flushes once per batch instead of
            once per file, since fsync-ing after every single file is most
            of the I/O cost at large scale and buys little durability a
            per-batch flush doesn't already give (worst case on a crash: redo
            one batch, same as the pre-existing per-file granularity already
            tolerated losing partial progress on a crash mid-write).
            """
            nonlocal processed, detections, failures_count, last_ui
            spath = str(path)
            if error is not None:
                failures_count += 1
            if file_dets:
                for d in file_dets:
                    writer.writerow(d)
                    counter[d.get('species', '')] += 1
                detections += len(file_dets)
            _copy_matches(path, relname, file_dets)
            manifest_f.write(json.dumps({'path': spath, 'ok': error is None,
                                         'error': error, 'n': len(file_dets)}) + '\n')
            processed += 1
            now = time.time()
            if now - last_ui > 0.3 or processed == total:
                last_ui = now
                summary = [{'species': s, 'count': c} for s, c in counter.most_common(50)]
                _set(job_id, processed=processed, detections=detections,
                     failed=failures_count, copied=len(copied_paths),
                     progress=int(processed / total * 100) if total else 100,
                     message=f'Processed {processed}/{total}', summary=summary)

        try:
            image_batch = []  # list of (path, relname) awaiting a batched detector call
            for path in all_files:
                spath = str(path)
                if spath in done_paths:
                    continue
                ext = path.suffix.lower()

                if ext in IMAGE_EXTS:
                    image_batch.append((path, _safe_rel(path, root)))
                    if len(image_batch) >= batch_size:
                        for p, relname, dets, error in _process_image_batch(detector, image_batch, run_dir, save_annotated):
                            _record(p, relname, dets, error)
                        image_batch = []
                        csv_f.flush()
                        manifest_f.flush()
                    continue

                # A non-image file ends the current run of batched images.
                if image_batch:
                    for p, relname, dets, error in _process_image_batch(detector, image_batch, run_dir, save_annotated):
                        _record(p, relname, dets, error)
                    image_batch = []
                    csv_f.flush()
                    manifest_f.flush()

                if ext in VIDEO_EXTS:
                    relname = _safe_rel(path, root)
                    file_dets, error = [], None
                    try:
                        file_dets = _process_video(detector, path, relname, run_dir, save_annotated, frame_stride)
                    except Exception as e:
                        error = str(e)
                    _record(path, relname, file_dets, error)
                    csv_f.flush()
                    manifest_f.flush()

            if image_batch:
                for p, relname, dets, error in _process_image_batch(detector, image_batch, run_dir, save_annotated):
                    _record(p, relname, dets, error)
        finally:
            csv_f.flush()
            manifest_f.flush()
            csv_f.close()
            manifest_f.close()
            if match_f:
                match_f.close()

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
             detections=detections, failed=failures_count, copied=len(copied_paths),
             csv=str(csv_path), failed_csv=str(fail_path),
             matched_csv=str(match_path) if copy_dir else '',
             summary=summary, completed=time.strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        try:
            (run_dir / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
        except Exception:
            pass
        _set(job_id, status='error', message=str(e), error=traceback.format_exc(),
             completed=time.strftime('%Y-%m-%d %H:%M:%S'))
