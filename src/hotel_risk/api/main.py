from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated
import time
import uuid

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hotel_risk.analytics import build_insights, predictions_to_frame
from hotel_risk.business import business_simulation, summarize_predictions
from hotel_risk.config import get_settings
from hotel_risk.db import BookingORM, PredictionORM, ScoringBatchORM, batch_to_dict, get_session, init_db
from hotel_risk.domain import DOMAIN_MODEL, LEAKAGE_COLUMNS, REQUIRED_COLUMNS, RISK_THRESHOLDS
from hotel_risk.features import CATEGORICAL_FEATURES, COLUMN_ALIASES, DEFAULT_VALUES, FEATURE_COLUMNS, NUMERIC_FEATURES, prepare_features, validate_input
from hotel_risk.logging_utils import setup_logging
from hotel_risk.ml import load_model, predict_dataframe
from hotel_risk.repository import create_scoring_batch, get_predictions, list_batches, persist_predictions
from hotel_risk.schemas import (
    BatchResponse,
    DomainModelResponse,
    HealthResponse,
    InsightResponse,
    ModelInfoResponse,
    OverviewResponse,
    SchemaResponse,
    ScoreRequest,
    ScoreResponse,
    SimulationRequest,
    SimulationResponse,
    ValidateRequest,
    ValidationResponse,
)

settings = get_settings()
logger = setup_logging("hotel_risk.api")
app = FastAPI(
    title="Hotel Booking Cancellation Risk API",
    version="0.9.0",
    description="REST API для ML-сервиса прогнозирования отмен гостиничных бронирований.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
    started = time.perf_counter()
    client = request.client.host if request.client else "unknown"
    logger.info(
        "request.start id=%s method=%s path=%s client=%s",
        request_id,
        request.method,
        request.url.path,
        client,
    )
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "request.error id=%s method=%s path=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["x-request-id"] = request_id
    logger.info(
        "request.end id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.on_event("startup")
def startup() -> None:
    init_db()
    artifact = load_model()
    logger.info("startup.complete env=%s db=%s model_path=%s model=%s log_file=%s", settings.environment, settings.database_url.split(":")[0], settings.model_path, artifact.get("metrics", {}).get("model", "unknown"), settings.log_file)


def _prediction_records(predictions: pd.DataFrame, limit: int = 200000) -> list[dict]:
    cols = [
        "booking_id",
        "hotel",
        "market_segment",
        "distribution_channel",
        "deposit_type",
        "adr",
        "total_nights",
        "booking_value",
        "risk_score",
        "cancellation_probability",
        "risk_category",
        "expected_loss",
        "business_priority_score",
        "deposit_loss_factor",
        "actionable",
        "financial_priority",
        "recommended_action",
        "top_factors",
    ]
    out = predictions[cols].head(limit).copy()
    out["booking_id"] = out["booking_id"].astype(str)
    return out.to_dict(orient="records")


def _validation_response(df: pd.DataFrame, strict: bool = False) -> ValidationResponse:
    report = validate_input(df, allow_defaults=not strict)
    try:
        prepared, _ = prepare_features(df, strict=strict)
        preview_cols = [
            "booking_id",
            "hotel",
            "arrival_date",
            "lead_time",
            "market_segment",
            "distribution_channel",
            "deposit_type",
            "adr",
            "total_nights",
            "booking_value",
        ]
        preview = prepared[[c for c in preview_cols if c in prepared.columns]].head(10).copy()
        if "arrival_date" in preview.columns:
            preview["arrival_date"] = preview["arrival_date"].astype(str)
        preview_records = preview.to_dict(orient="records")
    except Exception:
        preview_records = []
    return ValidationResponse(
        ok=report.ok,
        row_count=report.row_count,
        column_count=report.column_count,
        missing_required=report.missing_required,
        leakage_removed=report.leakage_removed,
        warnings=report.warnings,
        mapped_columns=report.mapped_columns or {},
        defaulted_columns=report.defaulted_columns or [],
        ignored_columns=report.ignored_columns or [],
        preview=preview_records,
    )


def _canonical_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "booking_id": "DEMO-001",
                "hotel": "City Hotel",
                "lead_time": 128,
                "arrival_date_year": 2017,
                "arrival_date_month": "August",
                "arrival_date_week_number": 32,
                "arrival_date_day_of_month": 12,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 2,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 1,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "deposit_type": "No Deposit",
                "customer_type": "Transient",
                "adr": 132.5,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
            },
            {
                "booking_id": "DEMO-002",
                "hotel": "Resort Hotel",
                "lead_time": 14,
                "arrival_date_year": 2017,
                "arrival_date_month": "September",
                "arrival_date_week_number": 37,
                "arrival_date_day_of_month": 15,
                "stays_in_weekend_nights": 0,
                "stays_in_week_nights": 3,
                "adults": 2,
                "children": 1,
                "babies": 0,
                "meal": "HB",
                "country": "FRA",
                "market_segment": "Direct",
                "distribution_channel": "Direct",
                "is_repeated_guest": 1,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 2,
                "reserved_room_type": "A",
                "deposit_type": "Non Refund",
                "customer_type": "Transient",
                "adr": 95.0,
                "required_car_parking_spaces": 1,
                "total_of_special_requests": 2,
            },
        ]
    )


def _flexible_sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "FLEX-001",
                "hotel_type": "City Hotel",
                "LeadTime": 95,
                "arrival_date": "2017-08-10",
                "no_of_weekend_nights": 1,
                "no_of_week_nights": 2,
                "no_of_adults": 2,
                "no_of_children": 0,
                "type_of_meal_plan": "BB",
                "market_segment_type": "Online TA",
                "booking_channel": "TA/TO",
                "room_type_reserved": "A",
                "avg_price_per_room": 125,
                "no_of_special_requests": 0,
                "random_comment": "ignored by the model",
            }
        ]
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    artifact = load_model()
    logger.debug("health model=%s", artifact.get("metrics", {}).get("model", "unknown"))
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        model=str(artifact.get("metrics", {}).get("model", "unknown")),
        database=settings.database_url.split(":")[0],
    )


@app.get("/api/v1/model-info", response_model=ModelInfoResponse, tags=["system"])
def model_info() -> ModelInfoResponse:
    artifact = load_model()
    metrics = artifact.get("metrics", {})
    return ModelInfoResponse(
        model_path=str(settings.model_path),
        model_name=str(metrics.get("model", "unknown")),
        metrics=metrics,
        feature_count=len(artifact.get("feature_columns", FEATURE_COLUMNS)),
        numeric_features=list(artifact.get("numeric_features", NUMERIC_FEATURES)),
        categorical_features=list(artifact.get("categorical_features", CATEGORICAL_FEATURES)),
        risk_thresholds={k.value: v for k, v in RISK_THRESHOLDS.items()},
    )


@app.get("/api/v1/domain-model", response_model=DomainModelResponse, tags=["domain"])
def domain_model() -> DomainModelResponse:
    return DomainModelResponse(
        entities=[{"name": e.name, "description": e.description} for e in DOMAIN_MODEL],
        relationships=[
            "HotelProperty 1—N ScoringBatch",
            "ScoringBatch 1—N Booking",
            "Booking 1—1 Prediction",
            "ScoringBatch 1—N ScoringJob",
            "AuditLog records all business-critical operations",
        ],
        storage="PostgreSQL in Docker Compose; SQLite fallback for local tests and one-command demo.",
        scaling="Redis queue + independently scalable model worker containers: docker compose up --scale worker=3.",
    )


@app.get("/api/v1/schema", response_model=SchemaResponse, tags=["schema"])
def schema() -> SchemaResponse:
    return SchemaResponse(
        required_columns=REQUIRED_COLUMNS,
        optional_columns=["booking_id", "arrival_date", "country", "is_canceled"],
        leakage_columns=LEAKAGE_COLUMNS,
        aliases=COLUMN_ALIASES,
        defaults=DEFAULT_VALUES,
        accepted_modes=["strict canonical CSV", "tolerant alias-based CSV", "JSON bookings payload"],
    )


@app.get("/api/v1/sample-csv", tags=["schema"])
def sample_csv(format: Annotated[str, Query(pattern="^(canonical|flexible)$")] = "canonical") -> StreamingResponse:
    df = _canonical_sample() if format == "canonical" else _flexible_sample()
    buf = StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=hotel_risk_{format}_sample.csv"},
    )


@app.post("/api/v1/validate", response_model=ValidationResponse, tags=["schema"])
def validate_json(payload: ValidateRequest, strict: bool = Query(default=False)) -> ValidationResponse:
    df = pd.DataFrame(payload.bookings)
    logger.info("validate_json rows=%s cols=%s strict=%s", len(df), df.shape[1], strict)
    result = _validation_response(df, strict=strict)
    logger.info("validate_json.done ok=%s missing=%s defaulted=%s warnings=%s", result.ok, len(result.missing_required), len(result.defaulted_columns), len(result.warnings))
    return result


@app.post("/api/v1/validate-csv", response_model=ValidationResponse, tags=["schema"])
async def validate_csv(file: UploadFile = File(...), strict: bool = Query(default=False)) -> ValidationResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Загрузите CSV-файл.")
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой.")
    logger.info("validate_csv file=%s bytes=%s strict=%s", file.filename, len(raw), strict)
    try:
        df = pd.read_csv(BytesIO(raw))
    except Exception as exc:
        logger.exception("validate_csv.read_failed file=%s", file.filename)
        raise HTTPException(status_code=400, detail=f"CSV не читается: {exc}") from exc
    result = _validation_response(df, strict=strict)
    logger.info("validate_csv.done rows=%s cols=%s ok=%s defaulted=%s ignored=%s warnings=%s", len(df), df.shape[1], result.ok, len(result.defaulted_columns), len(result.ignored_columns), len(result.warnings))
    return result


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["scoring"])
def score_json(payload: ScoreRequest, limit: Annotated[int, Query(ge=1, le=1000000)] = 1000000, strict: bool = Query(default=False)) -> ScoreResponse:
    df = pd.DataFrame(payload.bookings)
    logger.info("score_json rows=%s cols=%s limit=%s strict=%s", len(df), df.shape[1], limit, strict)
    try:
        predictions = predict_dataframe(df, strict=strict)
    except ValueError as exc:
        logger.warning("score_json.validation_failed error=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = summarize_predictions(predictions)
    logger.info("score_json.done rows=%s high_or_critical=%s expected_loss=%.2f", len(predictions), summary.get("high_or_critical"), float(summary.get("expected_loss_total", 0)))
    return ScoreResponse(summary=summary, predictions=_prediction_records(predictions, limit))


@app.post("/api/v1/score-csv", response_model=ScoreResponse, tags=["scoring"])
async def score_csv(file: UploadFile = File(...), limit: Annotated[int, Query(ge=1, le=1000000)] = 1000000, strict: bool = Query(default=False)) -> ScoreResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Загрузите CSV-файл.")
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой.")
    logger.info("score_csv file=%s bytes=%s limit=%s strict=%s", file.filename, len(raw), limit, strict)
    try:
        df = pd.read_csv(BytesIO(raw))
        predictions = predict_dataframe(df, strict=strict)
    except Exception as exc:
        logger.exception("score_csv.failed file=%s", file.filename)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    summary = summarize_predictions(predictions)
    logger.info("score_csv.done rows=%s high_or_critical=%s expected_loss=%.2f", len(predictions), summary.get("high_or_critical"), float(summary.get("expected_loss_total", 0)))
    return ScoreResponse(summary=summary, predictions=_prediction_records(predictions, limit))


@app.post("/api/v1/simulate", response_model=SimulationResponse, tags=["analytics"])
def simulate(payload: SimulationRequest) -> SimulationResponse:
    df = predictions_to_frame(payload.predictions)
    logger.info("simulate rows=%s top_share=%.2f success=%.2f cost=%.2f", len(df), payload.top_share, payload.intervention_success_rate, payload.cost_per_action)
    result = business_simulation(df, payload.top_share, payload.intervention_success_rate, payload.cost_per_action)
    return SimulationResponse(
        top_share=payload.top_share,
        intervention_success_rate=payload.intervention_success_rate,
        cost_per_action=payload.cost_per_action,
        result=dict(result),
    )


@app.post("/api/v1/batches/upload", response_model=BatchResponse, tags=["batches"])
async def upload_batch(
    db: Annotated[Session, Depends(get_session)],
    file: UploadFile = File(...),
    name: str | None = Query(default=None),
) -> BatchResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Загрузите CSV-файл.")
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Файл пустой.")
    if len(raw) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Файл слишком большой для синхронного MVP-скоринга.")
    logger.info("upload_batch.start file=%s bytes=%s name=%s", file.filename, len(raw), name)
    batch = create_scoring_batch(db, name or Path(file.filename).stem, source="csv", file_name=file.filename)
    try:
        df = pd.read_csv(BytesIO(raw))
        if len(df) > settings.sync_max_rows:
            raise ValueError(f"Слишком много строк для sync endpoint: {len(df)} > {settings.sync_max_rows}. Используйте async worker.")
        validation_report = validate_input(df, allow_defaults=True)
        predictions = predict_dataframe(df, strict=False)
        if validation_report.warnings:
            batch.error_message = " | ".join(validation_report.warnings)
        persist_predictions(db, batch, predictions)
        db.commit()
        logger.info("upload_batch.done batch_id=%s rows=%s high_risk=%s expected_loss=%.2f", batch.id, batch.row_count, batch.high_risk_count, batch.total_expected_loss)
    except Exception as exc:
        batch.status = "failed"
        batch.error_message = str(exc)
        db.commit()
        logger.exception("upload_batch.failed batch_id=%s error=%s", batch.id, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BatchResponse(**batch_to_dict(batch))


@app.get("/api/v1/batches", response_model=list[BatchResponse], tags=["batches"])
def batches(db: Annotated[Session, Depends(get_session)], limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[BatchResponse]:
    return [BatchResponse(**batch_to_dict(batch)) for batch in list_batches(db, limit=limit)]


@app.get("/api/v1/batches/{batch_id}", response_model=BatchResponse, tags=["batches"])
def batch_detail(batch_id: int, db: Annotated[Session, Depends(get_session)]) -> BatchResponse:
    batch = db.get(ScoringBatchORM, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    return BatchResponse(**batch_to_dict(batch))


@app.get("/api/v1/batches/{batch_id}/predictions", tags=["batches"])
def batch_predictions(
    batch_id: int,
    db: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=1000000)] = 1000000,
    category: str | None = None,
) -> dict:
    batch = db.get(ScoringBatchORM, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    preds = get_predictions(db, batch_id=batch_id, limit=limit, category=category)
    return {"batch": batch_to_dict(batch), "predictions": preds}


@app.get("/api/v1/batches/{batch_id}/insights", response_model=InsightResponse, tags=["analytics"])
def batch_insights(batch_id: int, db: Annotated[Session, Depends(get_session)]) -> InsightResponse:
    batch = db.get(ScoringBatchORM, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    preds = get_predictions(db, batch_id=batch_id, limit=1000000)
    insights = build_insights(predictions_to_frame(preds))
    return InsightResponse(**insights)


@app.get("/api/v1/batches/{batch_id}/export", tags=["batches"])
def batch_export_csv(batch_id: int, db: Annotated[Session, Depends(get_session)], category: str | None = None) -> StreamingResponse:
    batch = db.get(ScoringBatchORM, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    preds = get_predictions(db, batch_id=batch_id, limit=1000000, category=category)
    df = predictions_to_frame(preds)
    buf = StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    suffix = f"_{category}" if category else ""
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}{suffix}_predictions.csv"},
    )


@app.get("/api/v1/analytics/overview", response_model=OverviewResponse, tags=["analytics"])
def overview(db: Annotated[Session, Depends(get_session)]) -> OverviewResponse:
    batches_total = int(db.scalar(select(func.count(ScoringBatchORM.id))) or 0)
    bookings_scored_total = int(db.scalar(select(func.count(BookingORM.id))) or 0)
    high_risk_total = int(
        db.scalar(select(func.count(PredictionORM.id)).where(PredictionORM.risk_category.in_(["High", "Critical"]))) or 0
    )
    expected_loss_total = float(db.scalar(select(func.coalesce(func.sum(PredictionORM.expected_loss), 0.0))) or 0.0)
    recent = [BatchResponse(**batch_to_dict(batch)) for batch in list_batches(db, limit=10)]
    return OverviewResponse(
        batches_total=batches_total,
        bookings_scored_total=bookings_scored_total,
        high_risk_total=high_risk_total,
        expected_loss_total=expected_loss_total,
        recent_batches=recent,
    )


@app.post("/api/v1/jobs/score-file", tags=["workers"])
async def enqueue_file_job(file: UploadFile = File(...), name: str | None = Query(default=None)) -> dict:
    """Persist uploaded file and enqueue it for model workers. Requires Redis/RQ in Docker."""
    try:
        from redis import Redis
        from rq import Queue
        from hotel_risk.worker_jobs import score_file_job
    except Exception as exc:  # pragma: no cover - optional dependency
        raise HTTPException(status_code=503, detail="Redis/RQ dependencies are not installed in this runtime.") from exc
    raw = await file.read()
    logger.info("enqueue_file_job.start file=%s bytes=%s name=%s", file.filename, len(raw), name)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    path = settings.upload_dir / f"{int(pd.Timestamp.utcnow().timestamp())}_{file.filename}"
    path.write_bytes(raw)
    queue = Queue("hotel-risk-scoring", connection=Redis.from_url(settings.redis_url))
    job = queue.enqueue(score_file_job, str(path), name or Path(file.filename or "batch").stem, job_timeout="30m")
    logger.info("enqueue_file_job.done job_id=%s file_path=%s", job.id, path)
    return {"status": "queued", "queue": "hotel-risk-scoring", "job_id": job.id, "file_path": str(path)}
