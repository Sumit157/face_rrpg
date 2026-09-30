# Graph Report - rppg-deepfake-detector  (2026-09-28)

## Corpus Check
- Corpus is ~22,028 words - fits in a single context window. You may not need a graph.

## Summary
- 845 nodes · 1506 edges · 75 communities (52 shown, 23 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 47 edges (avg confidence: 0.9)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Web Frontend Components
- Face Detection MediaPipe
- ML Pipeline Orchestration
- Analysis UI Panels
- Web Frontend Dependencies
- Signal Feature Analysis
- API Analysis Handlers
- FastAPI Routes Schemas
- Harmonics Phase Features
- API Service Dependencies
- Confidence Feature Vector
- Express App Middleware
- rPPG Signal Methods
- Deployment Service Topology
- API Runtime Dependencies
- Server Config Bootstrap
- Inference Endpoint
- Monorepo Workspace Config
- Evidence And Tests
- Temporal HR Features
- Baseline Model Training
- TypeScript Base Config
- Prisma Data Services
- Analysis Job Queue
- Dataset Deep Training
- ML Service Rationale
- Frequency Domain Features
- Baseline Classifier
- API Dev Dependencies
- Web TSConfig
- Feature Extraction Logging
- Random Forest Training
- Signal Preprocessing Filtering
- Signal Quality Scoring
- Video Upload Handlers
- API TSConfig
- Project Limitations
- Package Scripts
- Feature Design Caveats
- Model Evaluation
- Shared Types Package
- Video Upload Middleware
- ML Config Setup
- Shared Types TSConfig
- Project Documentation
- API Pipeline Docs
- ROI Overlay Component
- 1D CNN Model
- Late Fusion Model
- TCN Model
- SVM Training
- XGBoost Baseline
- Face Overlay Component
- Video Player Component
- Data Storage Decisions
- Fusion Scope Notes
- CHROM Citation
- Detrending Citation
- Green Channel Citation
- POS Citation
- Model Download Script
- ML Service Root

## God Nodes (most connected - your core abstractions)
1. `run_analysis()` - 16 edges
2. `ApiError` - 15 edges
3. `analyse_regions()` - 15 edges
4. `extract_region_traces()` - 14 edges
5. `analyse_region()` - 13 edges
6. `BaselineClassifier` - 13 edges
7. `express` - 12 edges
8. `preprocess()` - 12 edges
9. `test_synchronised_regions_show_high_consistency()` - 12 edges
10. `apps/api Express service` - 12 edges

## Surprising Connections (you probably didn't know these)
- `rPPG Forensic Analyser web app entry point` --implements--> `apps/web React SPA`  [INFERRED]
  apps/web/index.html → docs/architecture.md
- `Props` --references--> `AnalysisProgressEvent`  [EXTRACTED]
  apps/web/src/components/analysis/AnalysisProgress.tsx → packages/shared-types/src/analysis.ts
- `main()` --calls--> `build_feature_vector()`  [EXTRACTED]
  scripts/extract_features.py → services/ml/app/features/feature_vector.py
- `main()` --uses--> `MotionResult`  [INFERRED]
  scripts/extract_features.py → services/ml/app/features/motion.py
- `main()` --calls--> `extract_region_traces()`  [EXTRACTED]
  scripts/preprocess_videos.py → services/ml/app/pipeline/extraction.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Docker Compose full-stack deployment** — docker_compose_postgres, docker_compose_redis, docker_compose_ml_service, docker_compose_api, docker_compose_worker, docker_compose_web [EXTRACTED 1.00]
- **Three rPPG extraction methods sharing one downstream preprocessing chain** — docs_methodology_green_method, docs_methodology_chrom_method, docs_methodology_pos_method, docs_methodology_downstream_filtering [EXTRACTED 1.00]
- **End-to-end analysis pipeline stages** — docs_architecture_request_flow, docs_methodology_signal_quality, docs_methodology_cross_region_consistency, docs_methodology_baseline_classifier, docs_methodology_explainability [EXTRACTED 1.00]

## Communities (75 total, 23 thin omitted)

### Community 0 - "Web Frontend Components"
Cohesion: 0.05
Nodes (53): AnalyseVideoParams, App(), Props, StatCard(), Layout(), LINKS, Sidebar(), Props (+45 more)

### Community 1 - "Face Detection MediaPipe"
Cohesion: 0.05
Nodes (41): mediapipe, mediapipe_tasks, mediapipe_tasks_python, RegionMask, FaceDetector, FaceLandmarks, ndarray, Face landmark detection using MediaPipe's Face Landmarker (Tasks API).… (+33 more)

### Community 2 - "ML Pipeline Orchestration"
Cohesion: 0.06
Nodes (30): cv2, ProgressCallback, pytest, AnalysisReport, Assembles the final structured analysis result returned to the caller (and, in…, Single function the FastAPI route (and CLI scripts / notebooks) call to run the…, run_analysis(), Frame enhancement for low-light video. Adapted from the rPPG deepfake detector… (+22 more)

### Community 3 - "Analysis UI Panels"
Cohesion: 0.07
Nodes (32): AnalysisProgress(), Props, STAGES, EvidencePanel(), Props, PredictionCard(), Props, VERDICT_STYLE (+24 more)

### Community 4 - "Web Frontend Dependencies"
Cohesion: 0.05
Nodes (39): dependencies, axios, echarts, echarts-for-react, react, react-dom, react-router-dom, @rppg/shared-types (+31 more)

### Community 5 - "Signal Feature Analysis"
Cohesion: 0.11
Nodes (27): main(), amplitude_cv(), ndarray, Coefficient of variation (std/mean) of each region's peak spectral amplitude…, ConsistencyResult, cross_region_consistency(), ndarray, analyse_region() (+19 more)

### Community 6 - "API Analysis Handlers"
Cohesion: 0.17
Nodes (13): createAnalysisHandler(), deleteAnalysisHandler(), getAnalysisFeaturesHandler(), getAnalysisHandler(), getAnalysisReportHandler(), getAnalysisSignalHandler(), listAnalysesHandler(), ApiError (+5 more)

### Community 7 - "FastAPI Routes Schemas"
Cohesion: 0.15
Nodes (16): fastapi, fastapi_middleware_cors, get, on_event, analyse(), post, POST /v1/analyse -- runs the full rPPG pipeline synchronously and returns the…, health() (+8 more)

### Community 8 - "Harmonics Phase Features"
Cohesion: 0.14
Nodes (17): scipy_signal, harmonic_ratio(), ndarray, Harmonic ratio: checks whether energy also appears at 2x and 3x the dominant…, power/freqs_hz should be the *full* spectrum (not band-limited), since 2x/3x…, instantaneous_phase(), phase_coherence(), ndarray (+9 more)

### Community 9 - "API Service Dependencies"
Cohesion: 0.11
Nodes (17): axios, @rppg/shared-types, typescript, name, private, version, ioredis, pino-http (+9 more)

### Community 10 - "Confidence Feature Vector"
Cohesion: 0.20
Nodes (12): dataclasses, calibrate(), CalibratedConfidence, Confidence calibration/annotation. Confidence is deliberately reported…, Pulls probability toward 0.5 in proportion to how far signal quality falls…, build_feature_vector(), FeatureVector, ndarray (+4 more)

### Community 11 - "Express App Middleware"
Cohesion: 0.17
Nodes (10): createApp(), errorHandler(), notFoundHandler(), analysisRouter, healthRouter, modelRouter, reportRouter, pythonService (+2 more)

### Community 12 - "rPPG Signal Methods"
Cohesion: 0.15
Nodes (12): numpy, chrom_signal(), ndarray, CHROM method (de Haan & Jeanne, 2013). Projects RGB onto two chrominance axes…, rgb_trace: shape (T, 3) with columns [R, G, B]. Returns shape (T,)., green_signal(), ndarray, Green-channel rPPG (Verkruysse et al., 2008). The simplest, most widely… (+4 more)

### Community 13 - "Deployment Service Topology"
Cohesion: 0.23
Nodes (16): rPPG Forensic Analyser web app entry point, API service container (docker-compose), ML service container (docker-compose), PostgreSQL 16 service (docker-compose), Redis 7 service (docker-compose), Web service container (docker-compose), Analysis worker container (docker-compose), Node API analysis/video endpoints (/api/videos, /api/analyses) (+8 more)

### Community 14 - "API Runtime Dependencies"
Cohesion: 0.13
Nodes (15): dependencies, axios, bullmq, cors, express, helmet, ioredis, multer (+7 more)

### Community 15 - "Server Config Bootstrap"
Cohesion: 0.21
Nodes (10): config, logger, app, httpServer, Express, Request, initWebSocketGateway(), ref_node_http (+2 more)

### Community 16 - "Inference Endpoint"
Cohesion: 0.21
Nodes (11): pydantic, inference(), InferenceRequest, InferenceResponse, BaseModel, post, POST /v1/inference -- run only the classifier stage on a precomputed feature…, PredictionResult (+3 more)

### Community 17 - "Monorepo Workspace Config"
Cohesion: 0.15
Nodes (12): name, packageManager, private, scripts, build, dev:api, dev:web, dev:worker (+4 more)

### Community 18 - "Evidence And Tests"
Cohesion: 0.24
Nodes (11): parametrize, build_evidence_and_warnings(), Generates human-readable evidence/warning strings from the computed features.…, pos_signal(), ndarray, rgb_trace: shape (T, 3) with columns [R, G, B]. Returns shape (T,). POS is…, make_synthetic_rgb(), Verifies the rPPG math recovers a known synthetic heart rate and that signal-… (+3 more)

### Community 19 - "Temporal HR Features"
Cohesion: 0.24
Nodes (9): Convenience aggregation of heart-rate-related features for one region., Time-domain and temporal-stability features across windows., temporal_stability(), TemporalFeatures, make_windows(), ndarray, Sliding-window segmentation over a preprocessed signal., Window (+1 more)

### Community 20 - "Baseline Model Training"
Cohesion: 0.23
Nodes (9): evaluate_group_cv(), load_dataset(), main(), Phase 8: train classical ML baselines (Logistic Regression, Random Forest, SVM)…, SVM training wrapper (secondary classical baseline for comparison)., sklearn_linear_model, sklearn_pipeline, sklearn_preprocessing (+1 more)

### Community 21 - "TypeScript Base Config"
Cohesion: 0.17
Nodes (11): compilerOptions, declaration, esModuleInterop, forceConsistentCasingInFileNames, lib, module, moduleResolution, resolveJsonModule (+3 more)

### Community 22 - "Prisma Data Services"
Cohesion: 0.25
Nodes (5): prisma, analysisQueue, RegisterVideoInput, videoService, @prisma/client

### Community 23 - "Analysis Job Queue"
Cohesion: 0.35
Nodes (8): ANALYSIS_QUEUE_NAME, AnalysisJobData, analysisWorker, processJob(), updateStatus(), redisConnection, emitAnalysisEvent(), bullmq

### Community 24 - "Dataset Deep Training"
Cohesion: 0.22
Nodes (8): argparse, json, Path, pathlib, iter_videos(), main(), Phase 7 dataset pipeline: extract per-region RGB traces for every video under…, Milestone 13: train the 1D CNN / TCN over multi-region rPPG sequences. This is…

### Community 25 - "ML Service Rationale"
Cohesion: 0.20
Nodes (11): POST /v1/inference ML endpoint, services/ml FastAPI rPPG service, Three-service split rationale, Face detection dependency on downloadable MediaPipe .task model, Haar cascade face detector fallback and CLAHE enhancement, fastapi==0.115.0 dependency, mediapipe==0.10.14 dependency, numpy==1.26.4 dependency (+3 more)

### Community 26 - "Frequency Domain Features"
Cohesion: 0.25
Nodes (8): scipy, Amplitude coefficient of variation across regions: real blood-volume pulses…, estimate_bpm(), ndarray, FFT / PSD spectrum and BPM estimation., Welch's method gives a smoother, lower-variance PSD estimate than a raw…, SpectrumResult, welch_psd()

### Community 27 - "Baseline Classifier"
Cohesion: 0.24
Nodes (6): BaselineClassifier, ndarray, Loads a trained sklearn model if present, otherwise falls back to a…, A simple, fully transparent weighted combination of the features most supported…, Even if harmonic_ratio is misleadingly high (the noise edge case above), a zero…, test_classifier_stays_uncertain_despite_misleading_harmonic_ratio()

### Community 28 - "API Dev Dependencies"
Cohesion: 0.20
Nodes (10): devDependencies, prisma, tsx, @types/cors, @types/express, @types/multer, @types/node, @types/uuid (+2 more)

### Community 29 - "Web TSConfig"
Cohesion: 0.20
Nodes (9): compilerOptions, jsx, lib, module, moduleResolution, noEmit, extends, include (+1 more)

### Community 30 - "Feature Extraction Logging"
Cohesion: 0.22
Nodes (7): itertools, Logger, Phase 7/8: turn the cached per-region RGB traces from preprocess_videos.py into…, get_logger(), Structured logging setup shared across the ML service., Cross-region physiological consistency. Real facial blood flow should pulse…, sys

### Community 31 - "Random Forest Training"
Cohesion: 0.22
Nodes (9): RandomForestClassifier, cross_validate(), ndarray, Random Forest training wrapper (Phase 8 classical ML baseline)., y convention: 1 = REAL, 0 = FAKE (must match BaselineClassifier.predict)., Subject-independent CV should be done via GroupKFold upstream (see…, train_random_forest(), sklearn_ensemble (+1 more)

### Community 32 - "Signal Preprocessing Filtering"
Cohesion: 0.33
Nodes (9): bandpass(), detrend(), normalise(), preprocess(), ndarray, Detrending, normalisation and band-pass filtering for rPPG traces., Remove slow drift (illumination change, minor head movement) using a…, Butterworth band-pass restricted to the physiologically plausible heart-rate… (+1 more)

### Community 33 - "Signal Quality Scoring"
Cohesion: 0.29
Nodes (8): assess_quality(), compute_snr(), ndarray, Signal-quality scoring. The single most important safety property of this…, Returns (snr_db, band_ratio) for the harmonic-aware SNR around peak_hz., Map SNR (dB) to an intuitive 0-100 score via a clipped linear map. Calibrated…, score_from_snr(), SignalQuality

### Community 34 - "Video Upload Handlers"
Cohesion: 0.42
Nodes (5): getVideoHandler(), uploadVideoHandler(), asyncHandler(), videoRouter, express

### Community 35 - "API TSConfig"
Cohesion: 0.22
Nodes (8): compilerOptions, module, moduleResolution, outDir, rootDir, extends, include, ../../tsconfig.base.json

### Community 36 - "Project Limitations"
Cohesion: 0.25
Nodes (9): Not implemented / out of scope, No cross-dataset generalisation testing performed, Signal-quality gating and confidence discounting are linear heuristics, Classifier is not validated (no labelled dataset), Baseline classifier with trained path and heuristic fallback, Explainability layer (evidence, warnings, confidence calibration), GroupKFold on subject ID for subject-independent evaluation, Heuristic fallback classifier (isHeuristicFallback) (+1 more)

### Community 37 - "Package Scripts"
Cohesion: 0.25
Nodes (8): scripts, build, dev, prisma:generate, prisma:migrate, start, test, worker

### Community 38 - "Feature Design Caveats"
Cohesion: 0.25
Nodes (8): Harmonic ratio feature unreliable without a real dominant peak, Motion, lighting and compression sensitivity of rPPG, Amplitude consistency (CV) feature, Cross-region physiological consistency check, Harmonic ratio feature, Phase coherence feature (Hilbert transform), rPPG-Based Deepfake Detector v2 notebook (Gauraav Kate & Atharav Shelar, MIT WPU), SNR-based signal quality scoring

### Community 39 - "Model Evaluation"
Cohesion: 0.25
Nodes (6): joblib, matplotlib, matplotlib_pyplot, pandas, Phase 8-9 / Milestone 15: evaluate a trained model on a held-out feature table…, sklearn_metrics

### Community 40 - "Shared Types Package"
Cohesion: 0.25
Nodes (7): main, name, private, scripts, typecheck, types, version

### Community 41 - "Video Upload Middleware"
Cohesion: 0.29
Nodes (5): storage, uploadVideo, multer, ref_node_crypto, ref_node_path

### Community 42 - "ML Config Setup"
Cohesion: 0.29
Nodes (4): os, Central configuration for the ML service. Values are overridable via…, Settings, Classical ML baseline: trains/loads a scikit-learn classifier over the feature…

### Community 43 - "Shared Types TSConfig"
Cohesion: 0.29
Nodes (6): compilerOptions, outDir, rootDir, extends, include, ../../tsconfig.base.json

### Community 44 - "Project Documentation"
Cohesion: 0.33
Nodes (6): API Reference (HTTP/WebSocket endpoints), Architecture document, Implementation status document, Limitations document, Methodology document, rPPG-Based Explainable Deepfake Detection (project)

### Community 45 - "API Pipeline Docs"
Cohesion: 0.33
Nodes (6): POST /v1/analyse ML endpoint, Why FastAPI progress is not streamed end-to-end, Request flow (video upload to result), Fully implemented and tested ML core, End-to-end rPPG pipeline (video to face/ROI to signal to features to classifier to explainability), pytest==8.3.3 testing dependency

### Community 46 - "ROI Overlay Component"
Cohesion: 0.40
Nodes (3): COLORS, Props, Region

### Community 47 - "1D CNN Model"
Cohesion: 0.40
Nodes (3): build_model(), __init__(), 1D CNN over multi-region rPPG sequences (Milestone 13). Optional deep-learning…

### Community 48 - "Late Fusion Model"
Cohesion: 0.50
Nodes (3): FusionResult, late_fusion(), Visual-only + rPPG-only late-fusion (Milestone 14 / Objective: fusion).…

### Community 49 - "TCN Model"
Cohesion: 0.40
Nodes (3): build_model(), __init__(), Temporal Convolutional Network alternative to the 1D CNN (Milestone 13). A…

### Community 50 - "SVM Training"
Cohesion: 0.50
Nodes (4): Pipeline, ndarray, Wrapped in a Pipeline so the scaler travels with the model at inference time --…, train_svm()

### Community 51 - "XGBoost Baseline"
Cohesion: 0.50
Nodes (3): ndarray, XGBoost baseline (only used if xgboost is installed and justified by results --…, train_xgboost()

## Knowledge Gaps
- **189 isolated node(s):** `name`, `version`, `private`, `dev`, `build` (+184 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 402 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `bullmq` connect `Analysis Job Queue` to `API Service Dependencies`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Why does `react` connect `Web Frontend Components` to `Analysis UI Panels`, `Web Frontend Dependencies`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `react-router-dom` connect `Web Frontend Components` to `Analysis UI Panels`, `Web Frontend Dependencies`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **What connects `name`, `version`, `private` to the rest of the system?**
  _189 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Web Frontend Components` be split into smaller, more focused modules?**
  _Cohesion score 0.05094905094905095 - nodes in this community are weakly interconnected._
- **Should `Face Detection MediaPipe` be split into smaller, more focused modules?**
  _Cohesion score 0.05023796932839767 - nodes in this community are weakly interconnected._
- **Should `ML Pipeline Orchestration` be split into smaller, more focused modules?**
  _Cohesion score 0.06105457909343201 - nodes in this community are weakly interconnected._