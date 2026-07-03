from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from .db import BookingORM, PredictionORM, ScoringBatchORM, audit
from .domain import BatchStatus, RiskCategory


def create_scoring_batch(session: Session, name: str, source: str = "csv", file_name: str | None = None) -> ScoringBatchORM:
    batch = ScoringBatchORM(name=name, source=source, file_name=file_name, status=BatchStatus.CREATED.value)
    session.add(batch)
    session.flush()
    audit(session, "batch_created", "ScoringBatch", batch.id, {"name": name, "source": source})
    return batch


def persist_predictions(session: Session, batch: ScoringBatchORM, predictions: pd.DataFrame) -> None:
    batch.status = BatchStatus.SCORING.value
    session.flush()
    high_mask = predictions["risk_category"].isin([RiskCategory.HIGH.value, RiskCategory.CRITICAL.value])
    batch.row_count = int(len(predictions))
    batch.high_risk_count = int(high_mask.sum())
    batch.total_booking_value = float(predictions["booking_value"].sum()) if "booking_value" in predictions else 0.0
    batch.total_expected_loss = float(predictions.loc[high_mask, "expected_loss"].sum()) if "expected_loss" in predictions else 0.0

    for _, row in predictions.iterrows():
        booking = BookingORM(
            batch_id=batch.id,
            external_booking_id=str(row.get("booking_id", "")),
            hotel=str(row.get("hotel", "Unknown")),
            arrival_date=str(row.get("arrival_date", ""))[:10],
            market_segment=str(row.get("market_segment", "Unknown")),
            distribution_channel=str(row.get("distribution_channel", "Unknown")),
            deposit_type=str(row.get("deposit_type", "Unknown")),
            customer_type=str(row.get("customer_type", "Unknown")),
            adr=float(row.get("adr", 0.0) or 0.0),
            total_nights=int(row.get("total_nights", 1) or 1),
            booking_value=float(row.get("booking_value", 0.0) or 0.0),
            raw_json=row.to_json(force_ascii=False, date_format="iso"),
        )
        session.add(booking)
        session.flush()
        pred = PredictionORM(
            booking_id=booking.id,
            risk_score=float(row.get("risk_score", 0.0)),
            cancellation_probability=float(row.get("cancellation_probability", row.get("risk_score", 0.0))),
            risk_category=str(row.get("risk_category", "Low")),
            expected_loss=float(row.get("expected_loss", 0.0)),
            business_priority_score=float(row.get("business_priority_score", 0.0)),
            recommended_action=str(row.get("recommended_action", "")),
            top_factors_json=json.dumps(row.get("top_factors", []), ensure_ascii=False),
        )
        session.add(pred)
    batch.status = BatchStatus.COMPLETED.value
    audit(session, "scoring_completed", "ScoringBatch", batch.id, {"rows": batch.row_count, "high_risk": batch.high_risk_count})


def list_batches(session: Session, limit: int = 50) -> list[ScoringBatchORM]:
    stmt = select(ScoringBatchORM).order_by(desc(ScoringBatchORM.created_at)).limit(limit)
    return list(session.execute(stmt).scalars())


def get_predictions(session: Session, batch_id: int, limit: int = 500, category: str | None = None) -> list[dict[str, Any]]:
    stmt = (
        select(BookingORM, PredictionORM)
        .join(PredictionORM, PredictionORM.booking_id == BookingORM.id)
        .where(BookingORM.batch_id == batch_id)
        .order_by(desc(PredictionORM.business_priority_score))
        .limit(limit)
    )
    if category:
        stmt = stmt.where(PredictionORM.risk_category == category)
    rows = session.execute(stmt).all()
    out = []
    for booking, pred in rows:
        out.append({
            "booking_db_id": booking.id,
            "booking_id": booking.external_booking_id,
            "hotel": booking.hotel,
            "arrival_date": booking.arrival_date,
            "market_segment": booking.market_segment,
            "distribution_channel": booking.distribution_channel,
            "deposit_type": booking.deposit_type,
            "customer_type": booking.customer_type,
            "adr": booking.adr,
            "total_nights": booking.total_nights,
            "booking_value": booking.booking_value,
            "risk_score": pred.risk_score,
            "cancellation_probability": pred.cancellation_probability,
            "risk_category": pred.risk_category,
            "expected_loss": pred.expected_loss,
            "business_priority_score": pred.business_priority_score,
            "recommended_action": pred.recommended_action,
            "top_factors": json.loads(pred.top_factors_json or "[]"),
        })
    return out
