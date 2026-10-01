# Graph Report - docs  (2026-10-01)

## Corpus Check
- 5 files · ~3,459 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 54 nodes · 68 edges · 11 communities (10 shown, 1 thin omitted)
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 25 edges (avg confidence: 0.84)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- rPPG Methods & Model
- Signal Quality Features
- Storage & Visualization
- E2E Stack & Face Detection
- Known Gaps & Validation
- API Service Endpoints
- Progress Reporting
- WebSocket Live Updates
- Local Queue Mode
- Confidence & Evidence
- Process Architecture

## God Nodes (most connected - your core abstractions)
1. `Model-free rPPG decision` - 7 edges
2. `In-process analysis queue (analysis.queue.ts)` - 5 edges
3. `SNR-based signal quality score` - 5 edges
4. `rPPG-Based Deepfake Detector v2 notebook (Gauraav Kate & Atharav Shelar, MIT WPU)` - 5 edges
5. `Analysis sub-resource endpoints (signal, features, report, visuals)` - 4 edges
6. `apps/api Express service` - 4 edges
7. `No validated real-video accuracy` - 4 edges
8. `Shared rPPG preprocessing chain` - 4 edges
9. `Node API public endpoints (apps/api)` - 3 edges
10. `services/ml FastAPI ML service` - 3 edges

## Surprising Connections (you probably didn't know these)
- `Motion, lighting and compression sensitivity` --semantically_similar_to--> `SNR-based signal quality score`  [INFERRED] [semantically similar]
  limitations.md → methodology.md
- `Verified end-to-end local stack` --semantically_similar_to--> `Codebase generation environment constraints`  [INFERRED] [semantically similar]
  implementation_status.md → limitations.md
- `Fully implemented and tested modules` --references--> `app/pipeline/end_to_end.py orchestration`  [INFERRED]
  implementation_status.md → architecture.md
- `Known gaps` --references--> `Complete but unverified source (Docker, scripts)`  [INFERRED]
  architecture.md → implementation_status.md
- `synthetic_unvalidated baseline.joblib classifier` --conceptually_related_to--> `No validated real-video accuracy`  [INFERRED]
  implementation_status.md → limitations.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three-process architecture: web, api and ml services** — docs_architecture_three_process_split, docs_architecture_apps_web, docs_architecture_apps_api, docs_architecture_services_ml [EXTRACTED 1.00]
- **Upload-to-result flow: queue, HTTP client, progress registry and ML pipeline** — docs_architecture_request_flow, docs_architecture_analysis_queue, docs_architecture_python_service, docs_architecture_progress_registry, docs_architecture_end_to_end_pipeline [EXTRACTED 1.00]
- **Three rPPG extraction methods sharing one preprocessing chain** — docs_methodology_green_method, docs_methodology_chrom_method, docs_methodology_pos_method, docs_methodology_shared_preprocessing [EXTRACTED 1.00]

## Communities (11 total, 1 thin omitted)

### Community 0 - "rPPG Methods & Model"
Cohesion: 0.24
Nodes (10): synthetic_unvalidated baseline.joblib classifier, CHROM rPPG method, de Haan & Jeanne 2013 (CHROM), Green rPPG method, Model-free rPPG decision, POS rPPG method (default), Shared rPPG preprocessing chain, Tarvainen et al. 2002 (smoothness-priors detrending) (+2 more)

### Community 1 - "Signal Quality Features"
Cohesion: 0.28
Nodes (9): Harmonic ratio unreliable without a real dominant peak, Motion, lighting and compression sensitivity, Amplitude consistency feature (amplitude_cv), CLAHE dark-frame contrast enhancement, Cross-region physiological consistency, Harmonic ratio feature, rPPG-Based Deepfake Detector v2 notebook (Gauraav Kate & Atharav Shelar, MIT WPU), Phase coherence feature (+1 more)

### Community 2 - "Storage & Visualization"
Cohesion: 0.40
Nodes (6): Analysis sub-resource endpoints (signal, features, report, visuals), Data storage decisions, Pipeline tab visualization trace, Signal artifact files (storage/signals/{analysis_id}.json), Visualization trace files (storage/visuals/{analysis_id}.json), Fully implemented and tested modules

### Community 3 - "E2E Stack & Face Detection"
Cohesion: 0.33
Nodes (6): Video upload to result request flow, Complete but unverified source (Docker, scripts), Verified end-to-end local stack, MediaPipe face detection dependency and fail-closed behaviour, Codebase generation environment constraints, Haar cascade face detector fallback

### Community 4 - "Known Gaps & Validation"
Cohesion: 0.50
Nodes (5): Known gaps, Not implemented / explicitly out of scope, No cross-dataset generalisation testing performed, No validated real-video accuracy, StratifiedGroupKFold training protocol

### Community 5 - "API Service Endpoints"
Cohesion: 0.67
Nodes (4): ML service internal endpoints (services/ml), Node API public endpoints (apps/api), apps/api Express service, services/ml FastAPI ML service

### Community 6 - "Progress Reporting"
Cohesion: 0.50
Nodes (4): analysis:progress events with stage history, app/pipeline/end_to_end.py orchestration, Polling-based progress reporting, app/core/progress.py in-memory progress registry

### Community 7 - "WebSocket Live Updates"
Cohesion: 0.67
Nodes (3): WebSocket analysis room events, apps/web React SPA, Single-process WebSocket fan-out

### Community 8 - "Local Queue Mode"
Cohesion: 0.67
Nodes (3): In-process analysis queue (analysis.queue.ts), python.service.ts ML HTTP client, Zero-infra local mode (SQLite + in-process queue)

### Community 9 - "Confidence & Evidence"
Cohesion: 0.67
Nodes (3): Signal-quality gating is a heuristic, Confidence discounting by measurement strength, Evidence and warning generation

## Knowledge Gaps
- **8 isolated node(s):** `Pipeline tab visualization trace`, `No cross-dataset generalisation testing performed`, `Phase coherence feature`, `Amplitude consistency feature (amplitude_cv)`, `Verkruysse et al. 2008 (remote PPG green channel)` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 10 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Verified end-to-end local stack` connect `E2E Stack & Face Detection` to `Storage & Visualization`?**
  _High betweenness centrality (0.489) - this node is a cross-community bridge._
- **Why does `Model-free rPPG decision` connect `rPPG Methods & Model` to `Signal Quality Features`, `Confidence & Evidence`?**
  _High betweenness centrality (0.347) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Model-free rPPG decision` (e.g. with `synthetic_unvalidated baseline.joblib classifier` and `Evidence and warning generation`) actually correct?**
  _`Model-free rPPG decision` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `SNR-based signal quality score` (e.g. with `Harmonic ratio unreliable without a real dominant peak` and `Motion, lighting and compression sensitivity`) actually correct?**
  _`SNR-based signal quality score` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `Analysis sub-resource endpoints (signal, features, report, visuals)` (e.g. with `Data storage decisions` and `Signal artifact files (storage/signals/{analysis_id}.json)`) actually correct?**
  _`Analysis sub-resource endpoints (signal, features, report, visuals)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Pipeline tab visualization trace`, `No cross-dataset generalisation testing performed`, `Phase coherence feature` to the rest of the system?**
  _8 weakly-connected nodes found - possible documentation gaps or missing edges._