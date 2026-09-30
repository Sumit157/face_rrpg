FROM python:3.11-slim

# OpenCV/MediaPipe runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Face landmarker model asset (requires network at build time)
RUN mkdir -p models_assets && \
    curl -L -o models_assets/face_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

ENV STORAGE_DIR=/storage
ENV FACE_LANDMARKER_MODEL_PATH=/service/models_assets/face_landmarker.task

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
