FROM python:3.9-slim

WORKDIR /app

# System deps:
# - libgl1, libglib2.0-0: OpenCV runtime deps
# - libsm6 libxext6 libxrender1: common OpenCV headless deps
# - espeak + libespeak1: for pyttsx3 (even if you don't show UI in Docker)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    espeak \
    libespeak1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create folders used in Docker runtime (safe even if env overrides)
RUN mkdir -p /data /tmp/uploads

# FastAPI on 80 (like your old dockerfile)
EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
