#!/usr/bin/env bash
# Downloads the MediaPipe Face Landmarker model asset required by
# services/ml/app/face/detector.py. Requires network access.
set -euo pipefail
DEST="services/ml/models_assets/face_landmarker.task"
URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

mkdir -p "$(dirname "$DEST")"
echo "Downloading MediaPipe Face Landmarker model to $DEST ..."
curl -L -o "$DEST" "$URL"
echo "Done."
