import os, shutil, json, csv, secrets, threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
from detector import available_providers, species_labels, IMAGE_EXTS, VIDEO_EXTS
from utils import (scan_inputs, scan_inputs_cached, invalidate_scan_cache, start_scan_refresher,
                   list_models, list_two_stage_pipelines, human_bytes, safe_copy_upload)
from job_runner import start_job, get_job, list_jobs, request_stop

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get('MODEL_DIR', BASE / 'models'))
INPUT_DIR = Path(os.environ.get('INPUT_DIR', BASE / 'input'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', BASE / 'output'))
RUNS_DIR = Path(os.environ.get('RUNS_DIR', BASE / 'runs'))
for p in [MODEL_DIR, INPUT_DIR, OUTPUT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Start keeping the input-folder scan warm immediately, before the first
# request ever arrives -- see utils.start_scan_refresher for why a
# request-time cache alone isn't enough on a large input folder.
start_scan_refresher(INPUT_DIR)

app = Flask(__name__)
# Use a stable secret if provided (so flash messages survive restarts / multiple workers),
# otherwise generate a random one per process instead of shipping a hardcoded value.
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH_MB', '4096')) * 1024 * 1024


def _safe_run_path(job_id, *parts):
    """Resolve a path inside RUNS_DIR/<job_id>, refusing anything that escapes it."""
    job_id = secure_filename(job_id)
    if not job_id:
        abort(400)
    base = (RUNS_DIR / job_id).resolve()
    target = (base / Path(*parts)).resolve() if parts else base
    if base != target and base not in target.parents:
        abort(404)
    return base, target


def _load_status(job_id):
    """Return live job state, falling back to the persisted status.json on disk."""
    job = get_job(job_id)
    if job:
        return job
    base, status_path = _safe_run_path(job_id, 'status.json')
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding='utf-8'))
        except Exception:
            return {}
    return {}


_EMPTY_SCAN = {'files': [], 'total': 0, 'images': 0, 'videos': 0, 'count': 0}


def context(need_scan=True):
    """Build the data every page's base template needs.

    need_scan=False skips the input-folder walk entirely for pages that never
    render it (runs/run_detail/settings/gallery) -- with a large input folder
    (tens of thousands of files) that walk is the single most expensive thing
    a page load can do, so routes that don't need it shouldn't pay for it.
    Routes that do need it get a briefly-cached result (see
    utils.scan_inputs_cached) so quick repeat navigation doesn't re-walk the
    folder on every click either.
    """
    scan = scan_inputs_cached(INPUT_DIR) if need_scan else _EMPTY_SCAN
    models = list_models(MODEL_DIR)
    two_stage = list_two_stage_pipelines(MODEL_DIR)
    providers = available_providers()
    gpu_ready = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
    jobs = list_jobs(RUNS_DIR, limit=8)
    return dict(scan=scan, models=models, two_stage=two_stage, providers=providers, gpu_ready=gpu_ready, jobs=jobs, human_bytes=human_bytes)


@app.route('/')
def dashboard():
    return render_template('dashboard.html', **context())


@app.route('/input', methods=['GET', 'POST'])
def input_manager():
    if request.method == 'POST':
        files = request.files.getlist('files')
        saved = 0
        for f in files:
            if not f or not f.filename:
                continue
            name = secure_filename(f.filename)
            if not name:
                continue
            ext = Path(name).suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                try:
                    safe_copy_upload(f, INPUT_DIR / name)
                    saved += 1
                except Exception:
                    pass
        invalidate_scan_cache(INPUT_DIR)
        flash(f'Saved {saved} files to input folder')
        return redirect(url_for('input_manager'))
    return render_template('input.html', **context())


@app.route('/clear-input', methods=['POST'])
def clear_input():
    if INPUT_DIR.exists():
        for p in INPUT_DIR.iterdir():
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except Exception:
                pass
    invalidate_scan_cache(INPUT_DIR)
    flash('Input folder cleared')
    return redirect(url_for('input_manager'))


def _model_config(form):
    """Build the model part of a run config from the Processing form fields.

    Returns (config, error). Model choices are only accepted if they appear in
    the server-side listings, which also keeps user input from escaping MODEL_DIR.
    """
    pipeline = form.get('pipeline', 'single_stage')
    if pipeline == 'two_stage':
        pipelines = {p['name']: p for p in list_two_stage_pipelines(MODEL_DIR)}
        chosen_pipeline = pipelines.get(form.get('two_stage_model', ''))
        if not chosen_pipeline:
            return None, 'Please select a valid two-stage detector'
        return {
            'pipeline': 'two_stage',
            'pipeline_name': chosen_pipeline['name'],
            'model_path': str(MODEL_DIR / chosen_pipeline['detector']),
            'classifier_path': str(MODEL_DIR / chosen_pipeline['classifier']),
            'classifier_classes_path': str(MODEL_DIR / chosen_pipeline['classes']) if chosen_pipeline['classes'] else '',
        }, None
    model = form.get('model')
    if not model or model not in list_models(MODEL_DIR):
        return None, 'Please select a valid ONNX model'

    # Class names come only from the model's own <model>.txt / .names, else
    # the names embedded in the model. A freely chosen or shared class file
    # could belong to another model and silently mislabel every box.
    classes_path = ''
    for ext in ('.txt', '.names'):
        same_stem = (MODEL_DIR / model).with_suffix(ext)
        if same_stem.exists():
            classes_path = same_stem
            break
    return {
        'pipeline': 'single_stage',
        'pipeline_name': Path(model).stem,
        'model_path': str(MODEL_DIR / model),
        'classes_path': str(classes_path),
    }, None


# (model/class file paths + mtimes) -> species list. Loading a model to read
# its classes takes a moment, so each model is read once per change on disk.
_SPECIES_CACHE = {}
_SPECIES_LOCK = threading.Lock()


@app.route('/api/model-species')
def api_model_species():
    model_config, error = _model_config(request.args)
    if error:
        return jsonify({'error': error}), 400
    paths = [v for k, v in sorted(model_config.items()) if k.endswith('_path') and v]
    key = tuple((p, Path(p).stat().st_mtime if Path(p).exists() else 0) for p in paths)
    with _SPECIES_LOCK:
        cached = _SPECIES_CACHE.get(key)
    if cached is None:
        try:
            cached = {'species': species_labels(model_config)}
        except Exception as e:  # noqa: BLE001 - shown to the user in the picker
            return jsonify({'error': f'Could not read the class list: {e}'}), 500
        with _SPECIES_LOCK:
            _SPECIES_CACHE[key] = cached
    return jsonify(cached)


@app.route('/processing', methods=['GET', 'POST'])
def processing():
    if request.method == 'POST':
        model_config, model_error = _model_config(request.form)
        if model_error:
            flash(model_error)
            return redirect(url_for('processing'))

        # Cached, not a fresh scan_inputs() call: on a large input folder, a
        # full walk can itself take well over a minute on some filesystems
        # (observed on a Windows Docker bind mount with ~21k files), which
        # would otherwise make clicking "Start detection run" hang for that
        # long before the job even begins. The brief cache window means a
        # file dropped into the folder by hand in the last few seconds
        # (outside this app's own upload/clear actions, which already
        # invalidate the cache immediately) could be missed by this run --
        # an acceptable trade for not blocking job start on a multi-minute scan.
        scan = scan_inputs_cached(INPUT_DIR)
        files = [f['path'] for f in scan['files']]
        if not files:
            flash('No input files found. Add files in the Input Manager first.')
            return redirect(url_for('input_manager'))

        def _num(name, default, cast, lo=None, hi=None):
            try:
                v = cast(request.form.get(name, default))
            except (TypeError, ValueError):
                v = cast(default)
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        # Optional copying of matching videos into OUTPUT_DIR/<folder>. The
        # folder is reduced to one safe name so it can't escape OUTPUT_DIR.
        copy_matches = request.form.get('copy_matches') == 'on'
        copy_config = {'copy_matches': False}
        if copy_matches:
            folder = secure_filename(request.form.get('copy_folder', '').strip()) or 'matched-videos'
            species = list(dict.fromkeys(s.strip() for s in request.form.getlist('copy_species') if s.strip()))
            copy_config = {
                'copy_matches': True,
                'copy_folder': folder,
                'copy_dir': str(OUTPUT_DIR / folder),
                'copy_species': species,
                'copy_min_conf': _num('copy_min_conf', 0.5, float, 0.0, 1.0),
                'copy_min_frames': _num('copy_min_frames', 1, int, 1),
                'copy_images': request.form.get('copy_images') == 'on',
            }

        config = {
            **model_config,
            **copy_config,
            'files': files,
            'input_root': str(INPUT_DIR),
            'runs_dir': str(RUNS_DIR),
            'conf': _num('conf', 0.25, float, 0.0, 1.0),
            'iou': _num('iou', 0.45, float, 0.0, 1.0),
            # 0 = auto: use the size the model was trained at.
            'input_size': _num('input_size', 0, int, 0, 4096),
            'batch_size': _num('batch_size', 16, int, 1, 256),
            'frame_stride': _num('frame_stride', 1, int, 1),
            'save_annotated': request.form.get('save_annotated') == 'on',
            'providers': os.environ.get('ORT_PROVIDERS', 'CUDAExecutionProvider,CPUExecutionProvider'),
        }
        job_id = start_job(config)
        return redirect(url_for('run_detail', job_id=job_id))
    return render_template('processing.html', **context())


@app.route('/runs')
def runs():
    return render_template('runs.html', **context(need_scan=False), all_jobs=list_jobs(RUNS_DIR))


@app.route('/runs/<job_id>')
def run_detail(job_id):
    job = _load_status(job_id)
    return render_template('run_detail.html', **context(need_scan=False), job=job or None, job_id=job_id)


@app.route('/api/jobs/<job_id>')
def api_job(job_id):
    job = _load_status(job_id)
    if not job:
        return jsonify({}), 404
    return jsonify(job)


@app.route('/download/<job_id>/<kind>')
def download(job_id, kind):
    job = _load_status(job_id)
    if not job:
        abort(404)
    base, _ = _safe_run_path(job_id)

    if kind == 'zip':
        # Built on demand (not during the job) so completion stays fast for large
        # runs. Cached after first creation.
        zip_path = base / 'results.zip'
        if not zip_path.exists():
            from utils import make_zip
            make_zip(base, zip_path)
        return send_file(zip_path, as_attachment=True)

    key = {'csv': 'csv', 'failed': 'failed_csv', 'matched': 'matched_csv'}.get(kind)
    if not key:
        abort(404)
    path = job.get(key)
    if kind == 'matched' and not path:
        # Available while the run is still going, not only once it completes.
        path = str(base / 'matched_files.csv')
    if not path or not Path(path).exists():
        abort(404)
    rp = Path(path).resolve()
    if base != rp and base not in rp.parents:
        abort(404)
    return send_file(rp, as_attachment=True)


@app.route('/runs/<job_id>/resume', methods=['POST'])
def resume_run(job_id):
    job = _load_status(job_id)
    if not job:
        abort(404)
    if job.get('status') not in ('error', 'stopped', 'finalizing', 'running'):
        flash('This run is not in a resumable state.')
        return redirect(url_for('run_detail', job_id=job_id))
    cfg = job.get('config')
    # status.json trims config['files'] to a count, so we need the original list.
    # It is recoverable by re-scanning the input root the job used.
    if not isinstance(cfg, dict):
        flash('Cannot resume: original run configuration is missing.')
        return redirect(url_for('run_detail', job_id=job_id))
    input_root = cfg.get('input_root') or str(INPUT_DIR)
    scan = scan_inputs(Path(input_root))
    files = [f['path'] for f in scan['files']]
    if not files:
        flash('Cannot resume: no input files found in the original input folder.')
        return redirect(url_for('run_detail', job_id=job_id))
    cfg = dict(cfg)
    cfg['files'] = files
    cfg['runs_dir'] = str(RUNS_DIR)
    start_job(cfg, resume_job_id=secure_filename(job_id))
    flash('Resuming run — already-processed files will be skipped.')
    return redirect(url_for('run_detail', job_id=job_id))


@app.route('/runs/<job_id>/stop', methods=['POST'])
def stop_run(job_id):
    _safe_run_path(job_id)
    job_id = secure_filename(job_id)
    if request_stop(job_id):
        flash('Stopping — the run will finish the file it is on, then stop. You can resume it later.')
    else:
        flash('This run is not running in this server process, so there is nothing to stop.')
    return redirect(url_for('run_detail', job_id=job_id))


@app.route('/gallery/<job_id>')
def gallery(job_id):
    base, ann = _safe_run_path(job_id, 'annotated')
    species_filter = request.args.get('species', '').strip()

    # Driven by detections.csv rather than walking the annotated folder: for
    # a large run that folder can hold thousands of images, and an unfiltered
    # directory walk pays for all of them every time regardless of how many
    # are actually shown. The CSV already has filename -> species for every
    # detection, so it doubles as the species-filter index for free, and is
    # typically far smaller than the full image count (no detection = no row).
    filename_species = {}  # relname -> set of species seen in that file
    order = []  # relnames in first-seen (= processing) order
    all_species = set()
    csv_path = base / 'detections.csv'
    if csv_path.is_file():
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                fn = row.get('filename') or ''
                if not fn:
                    continue
                sp = row.get('species') or ''
                if fn not in filename_species:
                    filename_species[fn] = set()
                    order.append(fn)
                if sp:
                    filename_species[fn].add(sp)
                    all_species.add(sp)

    relnames = [fn for fn in order if not species_filter or species_filter in filename_species[fn]]

    imgs = []
    for fn in relnames:
        stem_path = Path(fn)
        for suffix in ('.jpg', '.png'):
            candidate = ann / stem_path.with_suffix(suffix)
            if candidate.is_file():
                imgs.append(str(candidate.relative_to(base)))
                break
        if len(imgs) >= 300:
            break

    return render_template('gallery.html', **context(need_scan=False), job_id=job_id,
                           images=imgs, all_species=sorted(all_species),
                           species_filter=species_filter)


@app.route('/runs/<job_id>/file/<path:rel>')
def run_file(job_id, rel):
    base, target = _safe_run_path(job_id, rel)
    if not target.is_file():
        abort(404)
    return send_file(target)


@app.route('/settings')
def settings():
    return render_template('settings.html', **context(need_scan=False), model_dir=MODEL_DIR, input_dir=INPUT_DIR, runs_dir=RUNS_DIR)


if __name__ == '__main__':
    # Development entry point only. In the container we serve via waitress (see Dockerfile CMD).
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)
