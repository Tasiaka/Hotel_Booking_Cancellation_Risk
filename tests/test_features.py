from __future__ import annotations

import pandas as pd

from hotel_risk.features import FEATURE_COLUMNS, prepare_features


def sample_booking() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "booking_id": "B-1",
            "hotel": "City Hotel",
            "lead_time": 145,
            "arrival_date_year": 2017,
            "arrival_date_month": "August",
            "arrival_date_week_number": 32,
            "arrival_date_day_of_month": 10,
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
            "adr": 120.0,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0,
            "reservation_status": "Canceled",
            "reservation_status_date": "2017-07-01",
        }
    ])


def test_prepare_features_removes_leakage_and_engineers_values():
    prepared, report = prepare_features(sample_booking())
    assert report.ok
    assert set(report.leakage_removed) == {"reservation_status", "reservation_status_date"}
    assert "reservation_status" not in prepared.columns
    assert prepared.loc[0, "total_nights"] == 3
    assert prepared.loc[0, "booking_value"] == 360.0
    assert prepared.loc[0, "is_long_lead_booking"] == 1
    assert all(col in prepared.columns for col in FEATURE_COLUMNS)
