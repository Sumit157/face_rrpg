# API Reference

## Node API (apps/api) — public

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Service + ML-service health |
| POST | `/api/videos` | Upload a video (multipart, field name `video`) |
| GET | `/api/videos/:id` | Get video metadata |
| GET | `/api/videos/:id/stream` | Stream the stored file (HTTP Range supported, for seeking) |
| POST | `/api/analyses` | Start an analysis (`videoId`, `method`, `regions`, `windowSeconds`, `stepSeconds`) |
| GET | `/api/analyses` | List recent analyses |
| GET | `/api/analyses/:id` | Get analysis status/result |
| DELETE | `/api/analyses/:id` | Delete an analysis |
| GET | `/api/analyses/:id/signal` | Raw signal/consistency data for the Signal Explorer |
| GET | `/api/analyses/:id/features` | Flat feature vector |
| GET | `/api/analyses/:id/report` | Downloadable report payload |
| GET | `/api/analyses/:id/visuals` | Pipeline-tab trace (stage timeline, frame detections, windows); 404 if none |
| GET | `/api/models` | List available model versions |
| WS | `analysis:{id}` room, events `analysis:progress`, `analysis:{stage}` | Real-time progress |

## ML service (services/ml) — internal only, called by apps/api

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Model-loaded status |
| POST | `/v1/analyse` | Run the full pipeline on a video path, returns the structured `AnalysisReport` |
| GET | `/v1/analyse/{id}/progress` | Stage/percent progress for an in-flight analysis (polled by apps/api) |
| POST | `/v1/inference` | Run only the classifier on a precomputed feature dict |

See `packages/shared-types/src/` for the exact TypeScript request/response shapes, and
`services/ml/app/api/schemas/analysis.py` for the Python side.

Progress events (`analysis:progress`) include an optional `history` array
(`{ stage, at }[]`, seconds-relative) forwarded from the ML progress registry — live stage
durations for the UI. The web UI deep-links the analysis tabs: `/analysis/:id?tab=pipeline`.
