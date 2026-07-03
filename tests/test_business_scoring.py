from __future__ import annotations

import pandas as pd

from hotel_risk.business import business_simulation, summarize_predictions
from hotel_risk.ml import predict_dataframe


def test_rule_fallback_scoring_produces_business_priority():
    df = pd.DataFrame([
        {
            "booking_id": "A",
            "hotel": "City Hotel",
            "lead_time": 220,
            "arrival_date_year": 2017,
            "arrival_date_month": "August",
            "arrival_date_week_number": 32,
            "arrival_date_day_of_month": 10,
            "stays_in_weekend_nights": 1,
            "stays_in_week_nights": 3,
            "adults": 2,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Groups",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": 1,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "deposit_type": "No Deposit",
            "customer_type": "Transient",
            "adr": 100.0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
        }
    ])
    pred = predict_dataframe(df)
    assert pred.loc[0, "risk_score"] > 0.5
    assert pred.loc[0, "business_priority_score"] > 0
    assert pred.loc[0, "recommended_action"]
    summary = summarize_predictions(pred)
    assert summary["rows"] == 1
    sim = business_simulation(pred, top_share=1.0, intervention_success_rate=0.25, cost_per_action=0)
    assert sim["gross_protected_revenue"] > 0
