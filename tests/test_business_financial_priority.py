from __future__ import annotations

from hotel_risk.business import enrich_predictions
from hotel_risk.domain import BookingAction
from hotel_risk.features import prepare_features


def test_non_refund_high_probability_is_not_actionable_when_expected_loss_is_small(two_bookings_df):
    df = two_bookings_df.copy()
    df.loc[0, "deposit_type"] = "Non Refund"
    df.loc[0, "adr"] = 198.0
    df.loc[0, "stays_in_week_nights"] = 3
    df.loc[0, "stays_in_weekend_nights"] = 0
    prepared, _ = prepare_features(df)
    enriched = enrich_predictions(prepared, [0.90, 0.20], calibrated_probabilities=[0.90, 0.20])
    row = enriched[enriched["booking_id"].astype(str).eq("B-1")].iloc[0]
    assert row["deposit_loss_factor"] == 0.0
    assert row["expected_loss"] == 0.0
    assert row["actionable"] is False or row["actionable"] == False
    assert row["recommended_action"] == BookingAction.LOW_FINANCIAL_PRIORITY.value


def test_no_deposit_monetary_risk_outranks_non_refund_probability(two_bookings_df):
    df = two_bookings_df.copy()
    df.loc[0, "deposit_type"] = "Non Refund"
    df.loc[0, "adr"] = 198.0
    df.loc[0, "stays_in_week_nights"] = 3
    df.loc[1, "deposit_type"] = "No Deposit"
    df.loc[1, "adr"] = 500.0
    df.loc[1, "stays_in_week_nights"] = 3
    prepared, _ = prepare_features(df)
    enriched = enrich_predictions(prepared, [0.95, 0.45], calibrated_probabilities=[0.95, 0.45])
    assert str(enriched.iloc[0]["booking_id"]) == "B-2"
    assert enriched.iloc[0]["expected_loss"] > enriched.iloc[1]["expected_loss"]


def test_non_refund_is_not_manager_action_even_when_probability_is_high(two_bookings_df):
    df = two_bookings_df.copy()
    df.loc[0, "deposit_type"] = "Non Refund"
    df.loc[0, "adr"] = 5000.0
    df.loc[0, "stays_in_week_nights"] = 4
    prepared, _ = prepare_features(df)
    enriched = enrich_predictions(prepared, [0.95, 0.20], calibrated_probabilities=[0.95, 0.20])
    row = enriched[enriched["booking_id"].astype(str).eq("B-1")].iloc[0]
    assert row["deposit_type"] == "Non Refund"
    assert row["actionable"] is False or row["actionable"] == False
    assert row["financial_priority"] == "Не требует ручной проверки"
