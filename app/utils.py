from pathlib import Path
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


def scan_inputs(root):
    root = Path(root)
    files = []
    total = 0
    images = 0
    videos = 0
    if root.exists():
        for p in root.rglob('*'):
            try:
                if not p.is_file():
                    continue
                ext = p.suffix.lower()
                if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                    st = p.stat()
                    total += st.st_size
                    if ext in IMAGE_EXTS:
                        images += 1
                    if ext in VIDEO_EXTS:
                        videos += 1
                    files.append({'path': str(p), 'name': p.name, 'size': st.st_size, 'ext': ext})
            except OSError:
                continue
    files.sort(key=lambda f: f['path'])
    return {'files': files, 'total': total, 'images': images, 'videos': videos, 'count': len(files)}


def list_models(model_dir):
    p = Path(model_dir)
    p.mkdir(exist_ok=True, parents=True)
    return sorted([x.name for x in p.glob('*.onnx')])


def list_class_files(model_dir):
    """Return candidate class-name files (.txt / .names) in the model directory."""
    p = Path(model_dir)
    p.mkdir(exist_ok=True, parents=True)
    names = set()
    for pattern in ('*.txt', '*.names'):
        for x in p.glob(pattern):
            names.add(x.name)
    return sorted(names)


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
