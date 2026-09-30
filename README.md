# rPPG-Based Explainable Deepfake Detection

Analyses remote photoplethysmography (rPPG) signals extracted from facial video as an
explainable, physiological cue for deepfake detection research — measuring whether a face
video shows spatially consistent, physiologically plausible blood-flow signal across the
forehead and cheeks.

**This is not a standalone deepfake detector — read [`docs/limitations.md`](docs/limitations.md)
before using or presenting any output.** In particular: no trained classifier ships with this
repo (no labelled dataset was provided), so out of the box every prediction comes from a
clearly-labelled, unvalidated heuristic fallback (`isHeuristicFallback: true`).

## What's real vs. scaffolded

The rPPG science — video → face/ROI tracking → Green/CHROM/POS signal extraction → filtering →
FFT/BPM → signal quality → cross-region consistency → feature vector → classifier →
explainability — is **implemented and tested end-to-end** (`services/ml/`, see
`services/ml/tests/`). The Node/Express API and React frontend around it have also been run
and verified end-to-end locally (upload → queued job → progress → completed analysis with
signal/report endpoints; see [`docs/implementation_status.md`](docs/implementation_status.md)).

## Detection pipeline

Each frame is tracked with MediaPipe Face Mesh (with a local Haar fallback), segmented to skin
pixels in YCbCr, and sampled from the forehead plus both cheeks. GREEN, CHROM, and POS extractors
run for every clip; the strongest method is selected after window-level SNR checks. The pipeline
detrends and band-pass filters signals to 0.7–4 Hz, suppresses background illumination drift, and
uses landmark/optical-flow motion to discount unstable measurements.

The final decision fuses a calibrated Random Forest probability with transparent cues: bilateral
cheek correlation, cross-region frequency and phase agreement, SNR, heart-rate stability,
waveform repeatability, morphology, and compression-resistant spectral concentration. Windows
near the SNR floor are down-weighted instead of causing a blanket inconclusive result. `UNCERTAIN`
is reserved for a missing face or a truly unmeasurable video; a face-detectable video otherwise
returns Real or Fake with quality-calibrated confidence. The result page shows those quality-gate
and method diagnostics, and **Signal Explorer** provides the per-region waveform and spectrum.

`models/production/baseline.joblib` is trained on synthetic physiological-coherence data and is
useful as a lightweight fallback. For a production claim, retrain and evaluate it with labelled,
subject-disjoint real and manipulated video using `scripts/extract_features.py`,
`scripts/train_baseline.py`, and `scripts/evaluate.py`.

## Quick start (local, zero infrastructure)

No Docker, Postgres or Redis needed: SQLite + an in-process job queue.

```bash
# 1. JS dependencies + database (SQLite)
pnpm install
cp .env.example .env
pnpm exec prisma migrate dev --schema prisma/schema.prisma   # creates prisma/dev.db

# 2. ML service (Python)
cd services/ml
python -m venv venv
venv\Scripts\activate          # Windows — use `source venv/bin/activate` elsewhere
pip install -r requirements.txt
cd ../..
node scripts/ml.mjs dev        # uvicorn on http://localhost:8000

# 3. API + web (new terminals)
pnpm dev:api                   # http://localhost:3000
pnpm dev:web                   # http://localhost:5173

# Tests
node scripts/ml.mjs test       # 13 pytest tests (stubbed face detector)
pnpm typecheck                 # all TS packages
```

Optional sample footage (synthetic face video with a 72 bpm pulse):

```bash
venv/Scripts/python scripts/make_test_video.py   # writes tests/fixtures/synthetic_face_video.mp4
```

A Docker Compose topology (SQLite volume + nginx-proxied web) is also provided and is
**untested in this environment** — the local path above is the verified one.

## Repository layout

```
apps/
  web/            React + Vite + TypeScript + Tailwind frontend
  api/             Express + TypeScript API (routing, jobs, DB, WebSocket)
services/
  ml/             FastAPI + OpenCV + MediaPipe + NumPy/SciPy/scikit-learn — the rPPG pipeline
packages/
  shared-types/   TypeScript types shared between api and web
prisma/
  schema.prisma   Database schema
scripts/          Dataset preprocessing, feature extraction, training, evaluation CLIs
data/, models/, storage/, experiments/  Working directories (gitignored contents)
docs/             Architecture, methodology, limitations, API reference
```

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — service boundaries, request flow, data storage decisions
- [`docs/methodology.md`](docs/methodology.md) — the rPPG methods, signal quality, consistency, classifier, and why
- [`docs/limitations.md`](docs/limitations.md) — what this system cannot tell you, read before relying on it
- [`docs/implementation_status.md`](docs/implementation_status.md) — exactly what's tested vs. scaffolded, and how to run it
- [`docs/api.md`](docs/api.md) — HTTP/WebSocket API reference
