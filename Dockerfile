# -----------------------------------------------------------------------------
# Powerlifting Trainer Assistant API - container image
# -----------------------------------------------------------------------------
# Builds a slim Python image with OpenCV + MediaPipe runtime libraries and the
# FastAPI app (main.py). Webcam/OpenGL windows are disabled at runtime via
# RUNNING_IN_DOCKER=1 in compose/deploy; uploaded .mp4 analysis still works.
#
# Build:   docker build -t powerlifting-api .
# Run:     docker run -p 8000:8000 -e OPENAI_API_KEY=... powerlifting-api
# Compose: see docker-compose if present for env_file and volume mounts.
# -----------------------------------------------------------------------------

# Debian-based slim image (smaller than full python; we add only what OpenCV needs).
FROM python:3.9-slim

WORKDIR /app

# OpenCV (cv2) and GUI-related libs pulled in by wheels often expect these at runtime.
# espeak / libespeak: optional TTS stack used by pyttsx3 in headless scenarios.
# Cleaning apt lists keeps the layer smaller.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    espeak \
    libespeak1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer cache when only app code changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source and assets (respect .dockerignore to avoid .env, __pycache__, etc.).
COPY . .

# Writable dirs for SQLite/DB_PATH and UPLOAD_DIR defaults used by main.py.
RUN mkdir -p /data /tmp/uploads

# Default port for local docker run; platforms like Render set PORT at runtime (see CMD).
EXPOSE 8000

# Bind all interfaces; PORT is injected by many PaaS hosts, default 8000 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
