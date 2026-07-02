import os, shutil, json, secrets
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, abort
from werkzeug.utils import secure_filename
from detector import available_providers, IMAGE_EXTS, VIDEO_EXTS
from utils import scan_inputs, list_models, list_class_files, human_bytes, safe_copy_upload
from job_runner import start_job, get_job, list_jobs

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get('MODEL_DIR', BASE / 'models'))
INPUT_DIR = Path(os.environ.get('INPUT_DIR', BASE / 'input'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', BASE / 'output'))
RUNS_DIR = Path(os.environ.get('RUNS_DIR', BASE / 'runs'))
CLASSES_PATH = MODEL_DIR / 'classes.txt'
for p in [MODEL_DIR, INPUT_DIR, OUTPUT_DIR, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

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


def context():
    scan = scan_inputs(INPUT_DIR)
    models = list_models(MODEL_DIR)
    class_files = list_class_files(MODEL_DIR)
    providers = available_providers()
    gpu_ready = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
    jobs = list_jobs(RUNS_DIR)[:8]
    return dict(scan=scan, models=models, class_files=class_files, providers=providers, gpu_ready=gpu_ready, jobs=jobs, human_bytes=human_bytes)


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
    flash('Input folder cleared')
    return redirect(url_for('input_manager'))


@app.route('/processing', methods=['GET', 'POST'])
def processing():
    if request.method == 'POST':
        model = request.form.get('model')
        if not model or secure_filename(model) != model or not (MODEL_DIR / model).exists():
            flash('Please select a valid ONNX model')
            return redirect(url_for('processing'))
        scan = scan_inputs(INPUT_DIR)
        files = [f['path'] for f in scan['files']]
        if not files:
            flash('No input files found. Add files in the Input Manager first.')
            return redirect(url_for('input_manager'))

        # Resolve the classes file. Priority: explicit selection (validated) ->
        # a .txt sharing the model's stem -> the legacy models/classes.txt.
        classes_path = CLASSES_PATH
        chosen = request.form.get('classes')
        if chosen and secure_filename(chosen) == chosen and (MODEL_DIR / chosen).exists():
            classes_path = MODEL_DIR / chosen
        else:
            same_stem = MODEL_DIR / (Path(model).stem + '.txt')
            if same_stem.exists():
                classes_path = same_stem

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

        config = {
            'model_path': str(MODEL_DIR / model),
            'classes_path': str(classes_path),
            'files': files,
            'input_root': str(INPUT_DIR),
            'runs_dir': str(RUNS_DIR),
            'conf': _num('conf', 0.25, float, 0.0, 1.0),
            'iou': _num('iou', 0.45, float, 0.0, 1.0),
            'input_size': _num('input_size', 640, int, 32, 4096),
            'frame_stride': _num('frame_stride', 1, int, 1),
            'save_annotated': request.form.get('save_annotated') == 'on',
            'providers': os.environ.get('ORT_PROVIDERS', 'CUDAExecutionProvider,CPUExecutionProvider'),
        }
        job_id = start_job(config)
        return redirect(url_for('run_detail', job_id=job_id))
    return render_template('processing.html', **context())


@app.route('/runs')
def runs():
    return render_template('runs.html', **context(), all_jobs=list_jobs(RUNS_DIR))


@app.route('/runs/<job_id>')
def run_detail(job_id):
    job = _load_status(job_id)
    return render_template('run_detail.html', **context(), job=job or None, job_id=job_id)


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

    key = {'csv': 'csv', 'failed': 'failed_csv'}.get(kind)
    if not key:
        abort(404)
    path = job.get(key)
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
    if job.get('status') not in ('error', 'finalizing', 'running'):
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


@app.route('/gallery/<job_id>')
def gallery(job_id):
    base, ann = _safe_run_path(job_id, 'annotated')
    imgs = []
    if ann.exists():
        for p in ann.rglob('*'):
            if p.suffix.lower() in IMAGE_EXTS:
                imgs.append(str(p.relative_to(base)))
    return render_template('gallery.html', **context(), job_id=job_id, images=imgs[:300])


@app.route('/runs/<job_id>/file/<path:rel>')
def run_file(job_id, rel):
    base, target = _safe_run_path(job_id, rel)
    if not target.is_file():
        abort(404)
    return send_file(target)


@app.route('/settings')
def settings():
    return render_template('settings.html', **context(), model_dir=MODEL_DIR, input_dir=INPUT_DIR, runs_dir=RUNS_DIR, classes_path=CLASSES_PATH)


if __name__ == '__main__':
    # Development entry point only. In the container we serve via waitress (see Dockerfile CMD).
    app.run(host='0.0.0.0', port=8501, debug=False, threaded=True)
