from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from .domain import MIN_ACTION_EXPECTED_LOSS, RiskCategory, categorize_risk, deposit_loss_factor, recommended_action


def local_risk_factors(row: pd.Series, risk_score: float) -> list[str]:
    factors: list[str] = []
    if float(row.get("lead_time", 0) or 0) > 180:
        factors.append("очень длинная глубина бронирования >180 дней")
    elif float(row.get("lead_time", 0) or 0) > 90:
        factors.append("длинная глубина бронирования >90 дней")
    if str(row.get("deposit_type", "")) == "No Deposit":
        factors.append("нет депозита")
    if float(row.get("previous_cancellations", 0) or 0) > 0:
        factors.append("есть предыдущие отмены")
    if float(row.get("total_of_special_requests", 0) or 0) == 0:
        factors.append("нет специальных запросов")
    if str(row.get("market_segment", "")) in {"Online TA", "Groups"}:
        factors.append(f"рискованный сегмент продаж: {row.get('market_segment')}")
    if str(row.get("distribution_channel", "")) == "TA/TO":
        factors.append("канал TA/TO")
    if float(row.get("required_car_parking_spaces", 0) or 0) == 0:
        factors.append("нет запроса парковки")
    if float(row.get("is_repeated_guest", 0) or 0) == 0:
        factors.append("гость не является повторным")
    if risk_score >= 0.8 and not factors:
        factors.append("комбинация признаков похожа на отмененные бронирования в истории")
    return factors[:5]


def enrich_predictions(
    prepared: pd.DataFrame,
    risk_scores: np.ndarray | list[float],
    calibrated_probabilities: np.ndarray | list[float] | None = None,
) -> pd.DataFrame:
    """Add business fields to model output.

    `risk_score` is the raw model score used mainly for ranking.
    `cancellation_probability` is the calibrated probability used for money calculations.
    If no calibrator is available, calibrated probability falls back to the raw score.
    """
    out = prepared.copy()
    raw_score = np.clip(np.asarray(risk_scores, dtype=float), 0.0, 1.0)
    calibrated = raw_score if calibrated_probabilities is None else np.clip(np.asarray(calibrated_probabilities, dtype=float), 0.0, 1.0)
    if len(calibrated) != len(raw_score):
        raise ValueError("calibrated_probabilities must have the same length as risk_scores")

    out["risk_score"] = raw_score
    out["cancellation_probability"] = calibrated
    out["risk_category"] = [categorize_risk(float(s)).value for s in out["cancellation_probability"]]
    out["deposit_loss_factor"] = out["deposit_type"].map(lambda x: deposit_loss_factor(str(x))).astype(float)
    out["expected_loss"] = out["cancellation_probability"] * out["booking_value"].astype(float).clip(lower=0) * out["deposit_loss_factor"]
    out["business_priority_score"] = out["expected_loss"]
    risk_is_high = out["risk_category"].isin([RiskCategory.HIGH.value, RiskCategory.CRITICAL.value])
    # No manual removal by tariff type here. The model calculates cancellation
    # probability for every booking. The financial queue is derived only from
    # calibrated probability and expected monetary loss. Non Refund naturally
    # drops out because its expected_loss is zero.
    out["actionable"] = (risk_is_high & (out["expected_loss"] >= MIN_ACTION_EXPECTED_LOSS)).astype(bool)
    out["financial_priority"] = np.where(out["actionable"], "К проверке", "Не требует ручной проверки")
    out["recommended_action"] = [
        recommended_action(RiskCategory(cat), float(loss)) for cat, loss in zip(out["risk_category"], out["expected_loss"])
    ]
    out["top_factors"] = [local_risk_factors(row, float(row["cancellation_probability"])) for _, row in out.iterrows()]
    out = out.sort_values("business_priority_score", ascending=False).reset_index(drop=True)
    return out


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "rows": 0,
            "avg_risk_score": 0.0,
            "high_or_critical": 0,
            "expected_loss_total": 0.0,
            "booking_value_total": 0.0,
        }
    categories = predictions["risk_category"].value_counts().to_dict()
    high_mask = predictions["risk_category"].isin([RiskCategory.HIGH.value, RiskCategory.CRITICAL.value])
    actionable_mask = predictions.get("actionable", high_mask & (predictions["expected_loss"] >= MIN_ACTION_EXPECTED_LOSS)).astype(bool)
    return {
        "rows": int(len(predictions)),
        "avg_risk_score": float(predictions.get("cancellation_probability", predictions["risk_score"]).mean()),
        "high_or_critical": int(high_mask.sum()),
        "actionable_count": int(actionable_mask.sum()),
        "expected_loss_total": float(predictions["expected_loss"].sum()),
        "actionable_expected_loss_total": float(predictions.loc[actionable_mask, "expected_loss"].sum()),
        "booking_value_total": float(predictions["booking_value"].sum()),
        "category_counts": categories,
    }


def business_simulation(predictions: pd.DataFrame, top_share: float = 0.2, intervention_success_rate: float = 0.25, cost_per_action: float = 150.0) -> OrderedDict[str, float]:
    if predictions.empty:
        return OrderedDict([
            ("selected_bookings", 0),
            ("expected_loss", 0.0),
            ("gross_protected_revenue", 0.0),
            ("intervention_cost", 0.0),
            ("net_effect", 0.0),
        ])
    n = max(1, int(len(predictions) * top_share))
    top = predictions.sort_values("business_priority_score", ascending=False).head(n)
    expected_loss = float(top["expected_loss"].sum())
    cost = float(n * cost_per_action)
    protected = float(expected_loss * intervention_success_rate)
    return OrderedDict([
        ("selected_bookings", float(n)),
        ("expected_loss", expected_loss),
        ("gross_protected_revenue", protected),
        ("intervention_cost", cost),
        ("net_effect", protected - cost),
    ])
