FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg libglib2.0-0 libgl1 libsm6 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8501/ || exit 1

WORKDIR /app/app
# Runs as root by default so bind-mounted host volumes (./runs, ./input, ...) are
# always writable. To run unprivileged, set `user:` in docker-compose.yml to a UID
# that owns the mounted folders on the host.
CMD ["python3", "wsgi.py"]
