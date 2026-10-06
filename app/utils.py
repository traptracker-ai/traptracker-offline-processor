from pathlib import Path
import os
import threading
import time
import zipfile
import exifread
from detector import IMAGE_EXTS, VIDEO_EXTS


def human_bytes(n):
    n = float(n or 0)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _iter_entries(dirpath):
    """Recursively yield os.DirEntry objects under dirpath.

    Built on os.scandir rather than Path.rglob: scandir's DirEntry carries
    the OS's own directory-listing stat data, so entry.stat() is typically
    satisfied without an extra per-file syscall (on Windows in particular,
    the listing syscall already returns file size/attributes for every
    entry) -- unlike Path.rglob('*').stat(), which always re-stats each
    path from scratch. On a filesystem where every syscall is relatively
    expensive (e.g. a Docker Desktop bind mount into a Windows host), this
    is the difference between one expensive round trip per directory and
    one per file.
    """
    try:
        with os.scandir(dirpath) as it:
            entries = list(it)
    except OSError:
        return
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                yield from _iter_entries(entry.path)
            elif entry.is_file(follow_symlinks=False):
                yield entry
        except OSError:
            continue


def scan_inputs(root):
    root = Path(root)
    files = []
    total = 0
    images = 0
    videos = 0
    if root.exists():
        for entry in _iter_entries(root):
            ext = os.path.splitext(entry.name)[1].lower()
            if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            total += size
            if ext in IMAGE_EXTS:
                images += 1
            if ext in VIDEO_EXTS:
                videos += 1
            files.append({'path': entry.path, 'name': entry.name, 'size': size, 'ext': ext})
    files.sort(key=lambda f: f['path'])
    return {'files': files, 'total': total, 'images': images, 'videos': videos, 'count': len(files)}


# scan_inputs walks and stats every file in the input folder -- with a large
# (e.g. ~21k-file) input directory this alone can take several seconds, even
# with the os.scandir-based walk above, on a slow filesystem (observed: 5-12s
# on a Windows Docker Desktop bind mount). A short request-time TTL cache
# does not fix this in practice: at normal human navigation speed (seconds
# between clicks, reading each page first) almost every click is still a
# cache miss, so users still pay the full scan on nearly every page view.
# Instead, a single background thread keeps the result warm on its own
# schedule, so a request is *never* the one doing the slow walk -- it just
# reads whatever the refresher last computed, which is at most `interval`
# seconds stale. Explicit actions that actually change the folder (upload,
# clear) call invalidate_scan_cache(), which re-scans immediately rather
# than waiting for the next scheduled refresh.
_SCAN_CACHE = {}
_SCAN_REFRESHERS_STARTED = set()
_SCAN_REFRESH_LOCK = threading.Lock()


def _scan_refresh_loop(root, interval):
    key = str(root)
    while True:
        try:
            _SCAN_CACHE[key] = scan_inputs(root)
        except Exception:
            pass
        time.sleep(interval)


def start_scan_refresher(root, interval=20):
    """Start the background refresher for `root`. Safe to call more than
    once (e.g. module re-import) -- only the first call actually starts a
    thread. Call this once at app startup, not per-request."""
    key = str(root)
    with _SCAN_REFRESH_LOCK:
        if key in _SCAN_REFRESHERS_STARTED:
            return
        _SCAN_REFRESHERS_STARTED.add(key)
    t = threading.Thread(target=_scan_refresh_loop, args=(root, interval), daemon=True)
    t.start()


def scan_inputs_cached(root):
    """Read the background-refreshed result. Falls back to one direct,
    blocking scan only in the narrow window before the refresher (started
    at app startup) has completed its first pass."""
    hit = _SCAN_CACHE.get(str(root))
    return hit if hit is not None else scan_inputs(root)


def invalidate_scan_cache(root=None):
    """Re-scan immediately so the next request sees fresh data right away,
    rather than merely dropping the cache and leaving that request to pay
    for a cold re-scan itself."""
    keys = [str(root)] if root is not None else list(_SCAN_CACHE.keys())
    for key in keys:
        try:
            _SCAN_CACHE[key] = scan_inputs(key)
        except Exception:
            _SCAN_CACHE.pop(key, None)


SINGLE_STAGE_DIR = 'single-stage-detectors'
TWO_STAGE_DIR = 'two-stage-detectors'
STAGE1_SUFFIX = '_stage1_detector.onnx'
STAGE2_SUFFIX = '_stage2_classifier.onnx'


def _single_stage_dirs(model_dir):
    # Top-level models are still accepted so existing setups keep working.
    p = Path(model_dir)
    p.mkdir(exist_ok=True, parents=True)
    return [p, p / SINGLE_STAGE_DIR]


def list_models(model_dir):
    """Return single-stage ONNX models as paths relative to the model directory."""
    root = Path(model_dir)
    names = set()
    for d in _single_stage_dirs(root):
        for x in d.glob('*.onnx'):
            names.add(x.relative_to(root).as_posix())
    return sorted(names)


def list_class_files(model_dir):
    """Return candidate class-name files (.txt / .names) for single-stage models."""
    root = Path(model_dir)
    names = set()
    for d in _single_stage_dirs(root):
        for pattern in ('*.txt', '*.names'):
            for x in d.glob(pattern):
                names.add(x.relative_to(root).as_posix())
    return sorted(names)


def list_two_stage_pipelines(model_dir):
    """Pair <name>_stage1_detector.onnx with <name>_stage2_classifier.onnx (+ .txt)."""
    root = Path(model_dir)
    d = root / TWO_STAGE_DIR
    pipelines = []
    if not d.is_dir():
        return pipelines
    for det in sorted(d.glob('*' + STAGE1_SUFFIX)):
        name = det.name[:-len(STAGE1_SUFFIX)]
        clf = d / (name + STAGE2_SUFFIX)
        if not clf.exists():
            continue
        classes = clf.with_suffix('.txt')
        pipelines.append({
            'name': name,
            'detector': det.relative_to(root).as_posix(),
            'classifier': clf.relative_to(root).as_posix(),
            'classes': classes.relative_to(root).as_posix() if classes.exists() else '',
        })
    return pipelines


def exif_datetime(path):
    try:
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='EXIF DateTimeOriginal')
        for k in ['EXIF DateTimeOriginal', 'Image DateTime']:
            if k in tags:
                return str(tags[k])
    except Exception:
        pass
    return ''


def make_zip(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    zip_resolved = zip_path.resolve()
    # Write to a temp file first, then move into place, so the archive never
    # tries to include the partially written copy of itself.
    tmp_path = zip_path.with_suffix(zip_path.suffix + '.tmp')
    with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob('*'):
            if not p.is_file():
                continue
            rp = p.resolve()
            if rp == zip_resolved or rp == tmp_path.resolve():
                continue
            z.write(p, p.relative_to(src_dir))
    tmp_path.replace(zip_path)
    return zip_path


def safe_copy_upload(file_storage, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_storage.save(str(dest))
    return dest
