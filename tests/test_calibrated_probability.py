from __future__ import annotations

import numpy as np

from hotel_risk.business import enrich_predictions
from hotel_risk.features import prepare_features
from hotel_risk.ml import apply_probability_calibration, fit_probability_calibrator


def test_expected_loss_uses_calibrated_probability_not_raw_score(two_bookings_df):
    prepared, _ = prepare_features(two_bookings_df)
    enriched = enrich_predictions(prepared, risk_scores=[0.90, 0.90], calibrated_probabilities=[0.40, 0.10])
    first = enriched[enriched["booking_id"].astype(str) == "B-1"].iloc[0]
    assert first["risk_score"] == 0.90
    assert first["cancellation_probability"] == 0.40
    assert first["expected_loss"] == 0.40 * 360.0


def test_platt_calibrator_returns_probabilities_in_unit_interval():
    raw = np.array([0.05, 0.15, 0.35, 0.70, 0.90])
    y = np.array([0, 0, 0, 1, 1])
    calibrator = fit_probability_calibrator(raw, y)
    calibrated = apply_probability_calibration(raw, calibrator)
    assert calibrated.shape == raw.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))
