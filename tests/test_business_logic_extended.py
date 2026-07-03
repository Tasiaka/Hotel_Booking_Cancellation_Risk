from __future__ import annotations

import pandas as pd

from hotel_risk.business import business_simulation, enrich_predictions, local_risk_factors, summarize_predictions
from hotel_risk.domain import BookingAction, RiskCategory
from hotel_risk.features import prepare_features


def test_enrich_predictions_assigns_categories_actions_and_expected_loss(two_bookings_df):
    prepared, _ = prepare_features(two_bookings_df)
    enriched = enrich_predictions(prepared, [0.85, 0.20])
    critical = enriched[enriched["booking_id"].astype(str) == "B-1"].iloc[0]
    low = enriched[enriched["booking_id"].astype(str) == "B-2"].iloc[0]
    assert critical["risk_category"] == RiskCategory.CRITICAL.value
    assert critical["recommended_action"] == BookingAction.MANUAL_REVIEW.value
    assert critical["expected_loss"] == 0.85 * 360.0
    assert low["risk_category"] == RiskCategory.LOW.value
    assert low["deposit_loss_factor"] == 0.0


def test_enrich_predictions_sorts_by_business_priority_not_input_order(two_bookings_df):
    prepared, _ = prepare_features(two_bookings_df)
    enriched = enrich_predictions(prepared, [0.90, 0.95])
    assert str(enriched.iloc[0]["booking_id"]) == "B-1"
    assert enriched.iloc[0]["business_priority_score"] >= enriched.iloc[1]["business_priority_score"]


def test_local_risk_factors_limits_output_to_top_five(sample_booking_df):
    prepared, _ = prepare_features(sample_booking_df)
    factors = local_risk_factors(prepared.iloc[0], risk_score=0.9)
    assert 1 <= len(factors) <= 5
    assert any("глубина" in item for item in factors)
    assert any("депозит" in item for item in factors)


def test_summarize_predictions_empty_dataframe_has_zero_kpis():
    summary = summarize_predictions(pd.DataFrame())
    assert summary["rows"] == 0
    assert summary["expected_loss_total"] == 0.0
    assert summary["booking_value_total"] == 0.0


def test_business_simulation_applies_top_share_cost_and_success_rate(two_bookings_df):
    prepared, _ = prepare_features(two_bookings_df)
    enriched = enrich_predictions(prepared, [0.85, 0.20])
    sim = business_simulation(enriched, top_share=0.5, intervention_success_rate=0.25, cost_per_action=10)
    assert sim["selected_bookings"] == 1.0
    assert sim["expected_loss"] > 0
    assert sim["gross_protected_revenue"] == sim["expected_loss"] * 0.25
    assert sim["intervention_cost"] == 10.0
    assert sim["net_effect"] == sim["gross_protected_revenue"] - 10.0
