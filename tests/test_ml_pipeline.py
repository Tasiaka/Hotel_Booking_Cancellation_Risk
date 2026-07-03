from __future__ import annotations

import numpy as np
import pandas as pd

from hotel_risk.ml import RuleFallbackModel, load_model, predict_dataframe, topk_metrics


def test_rule_fallback_model_returns_probability_matrix(two_bookings_df):
    model = RuleFallbackModel()
    proba = model.predict_proba(two_bookings_df)
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] > proba[1, 1]


def test_predict_dataframe_accepts_explicit_model_artifact(two_bookings_df):
    pred = predict_dataframe(two_bookings_df, model_artifact={"pipeline": RuleFallbackModel(), "metrics": {}})
    assert len(pred) == 2
    assert pred["risk_score"].between(0, 1).all()
    assert {"risk_category", "expected_loss", "business_priority_score", "recommended_action"}.issubset(pred.columns)


def test_load_model_missing_path_returns_rule_fallback(tmp_path):
    artifact = load_model(tmp_path / "missing.joblib")
    assert isinstance(artifact["pipeline"], RuleFallbackModel)
    assert "rule fallback" in artifact["metrics"]["model"]


def test_topk_metrics_matches_manual_calculation():
    y_true = np.array([1, 0, 1, 0, 1])
    score = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    metrics = topk_metrics(y_true, score, k=0.4)
    assert metrics["precision@40"] == 0.5
    assert metrics["recall@40"] == 1 / 3
    assert round(metrics["lift@40"], 6) == round(0.5 / 0.6, 6)
