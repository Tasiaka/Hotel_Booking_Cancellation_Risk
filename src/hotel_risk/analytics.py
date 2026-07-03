from __future__ import annotations

from typing import Any
import pandas as pd

from .business import summarize_predictions


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def predictions_to_frame(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(predictions)
    for col in ["risk_score", "cancellation_probability", "booking_value", "expected_loss", "business_priority_score", "adr", "total_nights"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "cancellation_probability" not in df.columns and "risk_score" in df.columns:
        df["cancellation_probability"] = df["risk_score"]
    return df


def build_insights(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "summary": summarize_predictions(predictions),
            "risk_distribution": [],
            "segment_expected_loss": [],
            "deposit_expected_loss": [],
            "top_recommendations": [],
        }

    risk_distribution = (
        predictions.groupby("risk_category", dropna=False)
        .agg(count=("booking_id", "count"), expected_loss=("expected_loss", "sum"), avg_risk=("cancellation_probability", "mean"))
        .reset_index()
        .sort_values("expected_loss", ascending=False)
        .to_dict(orient="records")
    )
    segment_expected_loss = (
        predictions.groupby("market_segment", dropna=False)
        .agg(count=("booking_id", "count"), expected_loss=("expected_loss", "sum"), avg_risk=("cancellation_probability", "mean"))
        .reset_index()
        .sort_values("expected_loss", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    deposit_expected_loss = (
        predictions.groupby("deposit_type", dropna=False)
        .agg(count=("booking_id", "count"), expected_loss=("expected_loss", "sum"), avg_risk=("cancellation_probability", "mean"))
        .reset_index()
        .sort_values("expected_loss", ascending=False)
        .to_dict(orient="records")
    )
    top_recommendations = (
        predictions.groupby("recommended_action", dropna=False)
        .agg(count=("booking_id", "count"), expected_loss=("expected_loss", "sum"))
        .reset_index()
        .sort_values("expected_loss", ascending=False)
        .to_dict(orient="records")
    )
    return {
        "summary": summarize_predictions(predictions),
        "risk_distribution": risk_distribution,
        "segment_expected_loss": segment_expected_loss,
        "deposit_expected_loss": deposit_expected_loss,
        "top_recommendations": top_recommendations,
    }
