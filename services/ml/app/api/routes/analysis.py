"""POST /v1/analyse -- runs the full rPPG pipeline synchronously and
returns the structured result. The Node/Express layer queues the job and
streams progress to the browser; this endpoint stays a plain synchronous
call so it can also be exercised directly from a notebook, CLI script, or
curl for research/debugging.

While a call is in flight, GET /v1/analyse/{analysis_id}/progress reports
the pipeline's real stage/percentage (the same callback the CLI uses), so
the Node layer can poll and forward live progress over WebSocket.
"""
from fastapi import APIRouter, HTTPException

from app.api.schemas.analysis import AnalyseRequest, AnalyseResponse, ProgressResponse
from app.pipeline.end_to_end import run_analysis
from app.core import progress as progress_registry
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/v1/analyse", response_model=AnalyseResponse)
def analyse(request: AnalyseRequest) -> AnalyseResponse:
    progress_registry.start(request.analysis_id)

    def on_progress(stage: str, pct: int, message: str) -> None:
        progress_registry.update(request.analysis_id, stage, pct, message)

    try:
        report = run_analysis(
            video_path=request.video_path,
            method=request.method,
            regions=request.regions,
            window_seconds=request.window_seconds,
            step_seconds=request.step_seconds,
            analysis_id=request.analysis_id,
            on_progress=on_progress,
        )
    except FileNotFoundError as exc:
        # e.g. missing face-landmarker model asset -- surfaced as 503 so
        # the Node layer can retry/report a clear "service unavailable"
        # rather than a generic 500.
        progress_registry.update(
            request.analysis_id, "failed", 100, str(exc)
        )
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error during analysis %s", request.analysis_id)
        progress_registry.update(
            request.analysis_id, "failed", 100, f"Analysis failed: {exc}"
        )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    # run_analysis's final progress() call already marked completed/failed;
    # make sure the registry reflects the HTTP-level outcome either way.
    if not (progress_registry.get(request.analysis_id) or {}).get("done"):
        label = report.prediction.get("label") if isinstance(report.prediction, dict) else None
        stage = "failed" if label == "FAILED" else "completed"
        progress_registry.update(request.analysis_id, stage, 100, "Analysis finished")

    return AnalyseResponse(**report.to_dict())


@router.get("/v1/analyse/{analysis_id}/progress", response_model=ProgressResponse)
def analyse_progress(analysis_id: str) -> ProgressResponse:
    entry = progress_registry.get(analysis_id)
    if entry is None:
        # Not started yet (or already pruned) -- report a neutral queued
        # state rather than 404 so pollers don't log errors.
        return ProgressResponse(
            analysisId=analysis_id, stage="queued", progress=0,
            message="Waiting for analysis to start", done=False,
        )
    return ProgressResponse(**entry)
