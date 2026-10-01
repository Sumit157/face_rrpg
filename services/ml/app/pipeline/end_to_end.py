"""Single function the FastAPI route (and CLI scripts / notebooks) call to
run the complete video -> structured result pipeline.
"""
from __future__ import annotations
import time
import uuid
from typing import Callable, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.video.validation import validate_video
from app.pipeline.extraction import MAX_ANALYSED_SECONDS, extract_region_traces
from app.pipeline.analysis import analyse_regions
from app.pipeline.prediction import run_prediction
from app.explainability.evidence import build_evidence_and_warnings
from app.explainability.report import AnalysisReport
from app.explainability.artifacts import save_signal_artifacts
from app.explainability.visuals import save_visuals

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[str, int, str], None]]


def run_analysis(
    video_path: str,
    method: str = None,
    regions: List[str] = None,
    window_seconds: float = None,
    step_seconds: float = None,
    analysis_id: str = None,
    on_progress: ProgressCallback = None,
) -> AnalysisReport:
    method = method or settings.default_method
    regions = regions or settings.default_regions
    window_seconds = window_seconds or settings.default_window_seconds
    step_seconds = step_seconds or settings.default_step_seconds
    analysis_id = analysis_id or str(uuid.uuid4())

    t0 = time.time()

    # --- visualization trace state (written on success AND failure) ---
    timeline: List[dict] = []
    current_stage: dict = {"stage": None, "start": None, "message": None}
    frames: List[dict] = []
    extraction_info: dict = {}
    windows_info: List[dict] = []
    result_info: dict = {}

    def _now() -> float:
        return time.time() - t0

    def _close_stage() -> None:
        if current_stage["stage"] is None:
            return
        end = _now()
        timeline.append(
            {
                "stage": current_stage["stage"],
                "startS": round(current_stage["start"], 3),
                "endS": round(end, 3),
                "durationMs": int((end - current_stage["start"]) * 1000),
                "message": current_stage["message"],
            }
        )
        current_stage["stage"] = None

    def progress(stage: str, pct: int, message: str) -> None:
        if stage != current_stage["stage"]:
            _close_stage()
            current_stage["stage"] = stage
            current_stage["start"] = _now()
        current_stage["message"] = message
        logger.info("[%s] %s (%d%%) - %s", analysis_id, stage, pct, message)
        if on_progress:
            on_progress(stage, pct, message)

    try:
        progress("validating", 5, "Validating video file")
        validation = validate_video(video_path)
        warnings = list(validation.warnings)

        if not validation.valid:
            progress("failed", 100, "; ".join(validation.errors))
            return AnalysisReport(
                analysis_id=analysis_id,
                video_metadata={},
                face_quality={},
                signal_quality={},
                method=method,
                regions=regions,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                bpm={},
                consistency={},
                feature_vector={},
                prediction={"label": "FAILED", "reason": "; ".join(validation.errors)},
                confidence={},
                evidence=[],
                warnings=validation.errors,
            )

        progress("extracting_faces", 20, "Detecting face and tracking ROIs")
        extraction = extract_region_traces(video_path, regions)
        frames = extraction.frame_trace
        duration_s = extraction.frames_processed / extraction.fps if extraction.fps else 0.0
        extraction_info = {
            "fps": round(extraction.fps, 3),
            "framesProcessed": extraction.frames_processed,
            "durationSeconds": round(duration_s, 2),
            "sampleStride": extraction.frame_stride,
            "detectionRate": extraction.tracking.detection_rate,
            "detector": "haar" if extraction.used_haar_fallback else "mediapipe",
            "haarFallbackFrames": extraction.haar_frame_count,
            "capSeconds": MAX_ANALYSED_SECONDS,
            "enhancedFrames": sum(1 for e in frames if e.get("enhanced")),
        }

        if extraction.tracking.detection_rate < 0.3:
            warnings.append(
                "Face was detected in less than 30% of frames; results are unreliable."
            )

        progress("extracting_signals", 45, f"Extracting and comparing GREEN/POS/CHROM rPPG per region")
        progress("processing_signals", 60, "Filtering, quality-gating and windowing signals")
        progress("extracting_features", 75, "Computing temporal and cross-region features")
        region_analysis = analyse_regions(
            extraction.region_rgb, extraction.fps, method, window_seconds, step_seconds,
            extraction.face_centroids, extraction.flow_magnitudes,
        )

        if not region_analysis.usable_regions:
            warnings.append("No region produced a usable signal (face detection too sparse).")
            progress("failed", 100, "No usable signal extracted")
            return AnalysisReport(
                analysis_id=analysis_id,
                video_metadata=validation.metadata.to_dict() if validation.metadata else {},
                face_quality=extraction.tracking.to_dict(),
                signal_quality={},
                method=method,
                regions=regions,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                bpm={},
                consistency={},
                feature_vector={},
                prediction={"label": "FAILED", "reason": "No usable signal extracted"},
                confidence={},
                evidence=[],
                warnings=warnings,
            )

        # Per-window boundaries + BPM/quality/gate status for the Pipeline
        # tab's window-segmentation view (full-clip waveforms already live
        # in the signal artifacts).
        for region, res in region_analysis.region_results.items():
            kept_flags = region_analysis.gating.window_kept.get(region, [])
            entries = []
            for i, bpm_val in enumerate(res.window_bpm):
                start_s = i * step_seconds
                quality = res.window_quality[i] if i < len(res.window_quality) else None
                entries.append(
                    {
                        "index": i,
                        "startS": round(start_s, 2),
                        "endS": round(min(start_s + window_seconds, duration_s), 2),
                        "bpm": round(float(bpm_val), 2),
                        "quality": round(quality.quality_score, 1)
                        if quality is not None
                        else None,
                        "kept": kept_flags[i] if i < len(kept_flags) else True,
                    }
                )
            windows_info.append({"region": region, "windows": entries})

        progress("running_model", 88, "Fusing rPPG physiology and signal-quality evidence")
        bundle = run_prediction(
            region_analysis, detection_rate=extraction.tracking.detection_rate
        )

        # Persist large waveform/spectrum arrays as files (kept out of the
        # response body); the Node API serves them from GET /:id/signal.
        artifacts = save_signal_artifacts(
            analysis_id, region_analysis, extraction.fps, settings.storage_dir
        )

        evidence, extra_warnings = build_evidence_and_warnings(
            bundle.feature_vector.values, bundle.decision
        )
        warnings.extend(extra_warnings)

        bpm_summary = {
            region: res.temporal.to_dict() for region, res in region_analysis.region_results.items()
        }
        bpm_summary["overall"] = {
            "mean": round(bundle.feature_vector.values.get("bpm_mean", 0.0), 2),
            "std": round(bundle.feature_vector.values.get("bpm_std", 0.0), 2),
        }

        signal_quality_summary = {
            region: {
                "meanQualityScore": round(res.mean_quality_score, 1),
            }
            for region, res in region_analysis.region_results.items()
        }
        signal_quality_summary["overall"] = round(
            bundle.feature_vector.values.get("signal_quality_mean", 0.0), 1
        )
        signal_quality_summary["gating"] = region_analysis.gating.to_dict()
        signal_quality_summary["methodScores"] = {
            m: round(q, 1) for m, q in region_analysis.method_selection.mean_quality.items()
        }
        signal_quality_summary["selectedMethod"] = region_analysis.method_selection.selected
        signal_quality_summary["methodMeanBpm"] = {
            name: round(value, 1)
            for name, value in region_analysis.method_selection.mean_bpm.items()
        }
        signal_quality_summary["methodFrequencyAgreement"] = round(
            region_analysis.method_selection.frequency_agreement, 3
        )
        signal_quality_summary["measurementStrength"] = round(
            bundle.decision.measurement_strength, 3
        )

        result_info = {
            "method": method,
            "selectedMethod": region_analysis.method_selection.selected,
            "regions": region_analysis.usable_regions,
            "windowSeconds": window_seconds,
            "stepSeconds": step_seconds,
            "gating": region_analysis.gating.to_dict(),
            "cues": {k: round(v, 3) for k, v in bundle.decision.cues.items()},
        }

        progress("generating_report", 96, "Assembling structured result")
        report = AnalysisReport(
            analysis_id=analysis_id,
            video_metadata=validation.metadata.to_dict() if validation.metadata else {},
            face_quality=extraction.tracking.to_dict(),
            signal_quality=signal_quality_summary,
            method=method,
            regions=region_analysis.usable_regions,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            bpm=bpm_summary,
            consistency=region_analysis.consistency.to_dict(),
            feature_vector=bundle.feature_vector.to_dict(),
            prediction=bundle.prediction.to_dict(),
            confidence=bundle.confidence.to_dict(),
            evidence=evidence,
            warnings=warnings,
            artifacts=artifacts,
        )

        elapsed = time.time() - t0
        progress("completed", 100, f"Analysis complete in {elapsed:.1f}s")
        return report
    finally:
        # Runs on every exit path (success, validation failure, exception)
        # so the UI can always show how far the analysis got.
        _close_stage()
        try:
            save_visuals(
                analysis_id,
                settings.storage_dir,
                stages=timeline,
                extraction=extraction_info,
                frames=frames,
                windows=windows_info,
                result=result_info,
            )
        except Exception:  # noqa: BLE001 -- visualization must never break analysis
            logger.exception("Failed to save visuals trace for %s", analysis_id)
