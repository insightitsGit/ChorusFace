# ChorusFace face service — containerized TickFeed avatar + FaceBridge + MJPEG embed.
# Build from repo root (world must already be baked under output/worlds/tickfeed).
#
#   docker build -t chorusface-face .
#   docker run --rm -p 8766:8766 -e CHORUSFACE_BRIDGE_TOKEN=chorusface-beta chorusface-face
#
# GPU (Linux + NVIDIA): use compose profile `gpu` (see docker-compose.yml).

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODERNGL_WINDOW=headless \
    CHORUSFACE_PRODUCT_BETA=1 \
    CHORUSFACE_HEADLESS_SERVICE=1 \
    CHORUSFACE_BRIDGE_HOST=0.0.0.0 \
    CHORUSFACE_BRIDGE_PORT=8766 \
    CHORUSFACE_BRIDGE_TOKEN=chorusface-beta \
    CHORUSFACE_BRIDGE_CORS=* \
    CHORUSFACE_STREAM_FPS=12 \
    CHORUSFACE_WORLD=/app/output/worlds/tickfeed

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        libosmesa6 \
        mesa-utils \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY connectors ./connectors
COPY output/worlds/tickfeed ./output/worlds/tickfeed

RUN pip install --no-cache-dir -e ".[voice,ml]"

EXPOSE 8766

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/health', timeout=3)"

CMD ["python", "scripts/run_chorusface_service.py"]
