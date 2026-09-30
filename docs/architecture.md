# Architecture

## Overview

Three processes, each with a single clear responsibility (zero-infra local mode: SQLite +
an in-process job queue — no Docker, Postgres or Redis required):

```
apps/web (React/Vite)  --HTTP/WebSocket-->  apps/api (Express)  --HTTP-->  services/ml (FastAPI)
                                                   |                              |
                                                   v                              v
                                            SQLite (Prisma)               OpenCV/MediaPipe/
                                            prisma/dev.db                 NumPy/SciPy/sklearn
                                                   |
                                                   v
                                        in-process analysis queue
                                        (apps/api/src/jobs/analysis.queue.ts)
```

- **apps/web** — React SPA. Uploads video, configures an analysis run, subscribes to
  progress, renders results and signal visualisations. Contains zero signal-processing
  logic; it only ever displays what the API returns.
- **apps/api** — Express + TypeScript. Owns video storage bookkeeping, the analysis queue
  (in-process, concurrency 2, retry with exponential backoff), the SQLite schema (Prisma),
  and WebSocket progress events. Contains zero OpenCV/ML code — its only contact with the
  science is a single HTTP client (`services/python.service.ts`) that calls the Python
  service.
- **services/ml** — FastAPI + OpenCV + MediaPipe + NumPy/SciPy/scikit-learn. Owns 100% of
  the actual video processing, rPPG extraction, feature engineering, and classification.
  Stateless: given a video path and parameters, it returns a structured JSON result. It
  does not know about SQLite or HTTP sessions.

This split exists so the computationally/scientifically hard part (rPPG extraction) can be
developed, tested, and iterated on entirely independently of the web application shell — see
`services/ml/tests/`, which run with zero Node dependencies.

## Request flow (video upload → result)

1. `POST /api/videos` (apps/api) — multer saves the file under `storage/uploads/` with a
   server-generated filename; a `Video` row is created. Paths are stored absolute so the
   API can be started from any working directory.
2. `POST /api/analyses` (apps/api) — validates params (Zod), creates an `Analysis` row
   (`status: "queued"`), pushes the job onto the in-process queue (`analysis.queue.ts`).
3. The queue worker (same Node process, concurrency 2) sets
   `status: "extracting_faces"` and calls `POST /v1/analyse` on the ML service. While that
   blocking call is in flight, `python.service.ts` polls
   `GET /v1/analyse/:id/progress` every 700 ms and persists each non-terminal stage
   (`extracting_signals` → `processing_signals` → … → `generating_report`) to the
   `Analysis` row, so the frontend sees granular progress.
4. `services/ml` runs `app/pipeline/end_to_end.py`: validate → extract faces/ROIs → extract
   rPPG per region → filter/window → FFT/BPM → signal quality → cross-region consistency →
   feature vector → classifier → calibrated confidence → evidence/warnings → structured
   report. The in-memory progress registry (`app/core/progress.py`) backs the progress
   endpoint; per-region waveform/spectrum arrays are written to
   `storage/signals/{analysis_id}.json` (`app/explainability/artifacts.py`).
5. On success the queue persists the JSON result (`Analysis.result`, a JSON-encoded text
   column — SQLite has no native Json type), sets `status: "completed"`, and emits a
   WebSocket event. Failed runs (including the ML service's `FAILED` verdict marker) set
   `status: "failed"` plus `errorMessage`; crashed jobs are retried up to 3 times with 5 s
   exponential backoff before being marked failed.
6. `apps/web` polls `GET /api/analyses/:id` and/or listens on the WebSocket channel and
   renders the result once `status: "completed"`.

## Progress: implemented via polling (not streaming)

`POST /v1/analyse` is a single blocking call, but the ML service records fine-grained stage
updates in `app/core/progress.py`, which the Node side polls during the call. This keeps
the transport simple (plain HTTP, no SSE/WebSocket fan-in between services) while still
driving the per-stage progress bar. `app/pipeline/end_to_end.py`'s `ProgressCallback` type
is the seam if streaming is ever wanted.

## Pipeline visualization (Analysis page "Pipeline" tab)

While an analysis runs, `app/pipeline/extraction.py` records a normalised per-frame trace
(face box + ROI polygons in 0-1 coordinates, detector used, detection/missed) sampled at
~5 Hz (`FrameTraceCollector` in `app/explainability/visuals.py`, bounded to 600 entries).
`app/pipeline/end_to_end.py` closes it out with the stage timeline, extraction stats and
per-window segmentation, and writes `storage/visuals/{analysis_id}.json` in a `finally`
block — so even failed analyses show how far they got.

- `GET /api/analyses/:id/visuals` reads that file (404 if the analysis predates tracing).
- `GET /api/videos/:id/stream` streams the stored upload with HTTP Range support
  (required for seeking); storage is never served statically.
- The React `PipelineView` overlays the trace on the video synchronised to the playhead
  (nearest sample, missed-frame chips, detection strip) and renders the stage timeline,
  extraction stats and sliding-window cards. The tab is deep-linkable:
  `/analysis/:id?tab=pipeline`.

## WebSocket fan-out

The queue runs **in the same process** as the HTTP server, so
`websocket/analysis.gateway.ts`'s `emitAnalysisEvent` always reaches every connected
browser — no cross-process adapter needed. This is a deliberate consequence of the
zero-infra mode: a future multi-instance deployment would need the `socket.io-redis`
adapter (or equivalent pub/sub) plus an external queue.

## Data storage decisions

- Raw video files live on disk (`storage/uploads/`, absolute paths in the DB), never in
  the database.
- Full per-analysis structured results are stored as a single JSON-encoded text column
  (`Analysis.result`), parsed into a flat `AnalysisRecord` on the way out
  (`apps/api/src/services/serializers.ts`) — the shape is still actively evolving during
  research (spec section 16). `regions` is likewise JSON-encoded text (SQLite has no
  array type). `RPPGResult`/`Feature` rows exist for future querying once the schema
  stabilises.
- Large arrays (raw waveforms, spectra) are **not** embedded in the result: the ML service
  writes them to `storage/signals/{analysis_id}.json`, the result references them via the
  `artifacts` dict, and `GET /api/analyses/:id/signal` reads the file back for the Signal
  Explorer UI.
- Same pattern for the visualization trace: `storage/visuals/{analysis_id}.json` (frame
  trace + stage timeline), served on demand by `GET /api/analyses/:id/visuals`.
- All relative storage/model paths (Node `STORAGE_DIR`, Python `STORAGE_DIR`/`MODEL_PATH`)
  are resolved against the repo root by walking up to `pnpm-workspace.yaml`, so every
  service agrees on one storage tree regardless of the working directory it was started
  from.

## Known gaps (see docs/implementation_status.md for the full list)

- Docker artifacts exist but are untested in this environment (Docker unavailable).
- No trained classifier ships with the repo; predictions come from the clearly-labelled
  heuristic fallback until `scripts/train_baseline.py` + `scripts/evaluate.py` are run on
  a real labelled dataset.
