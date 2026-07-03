from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("HOTEL_RISK_DATABASE_URL", "sqlite:///./test_hotel_risk.db")
os.environ.setdefault("HOTEL_RISK_MODEL_PATH", "models/__missing_test_model.joblib")
os.environ.setdefault("HOTEL_RISK_METRICS_PATH", "reports/tables/__test_metrics.json")
os.environ.setdefault("HOTEL_RISK_UPLOAD_DIR", "storage/test_uploads")


@pytest.fixture(autouse=True)
def _test_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOTEL_RISK_UPLOAD_DIR", str(tmp_path / "uploads"))


@pytest.fixture
def sample_booking_dict() -> dict:
    return {
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
    }


@pytest.fixture
def sample_booking_df(sample_booking_dict) -> pd.DataFrame:
    return pd.DataFrame([sample_booking_dict])


@pytest.fixture
def two_bookings_df(sample_booking_dict) -> pd.DataFrame:
    low = dict(sample_booking_dict)
    low.update(
        {
            "booking_id": "B-2",
            "hotel": "Resort Hotel",
            "lead_time": 3,
            "market_segment": "Direct",
            "distribution_channel": "Direct",
            "is_repeated_guest": 1,
            "previous_cancellations": 0,
            "deposit_type": "Non Refund",
            "adr": 80.0,
            "required_car_parking_spaces": 1,
            "total_of_special_requests": 2,
        }
    )
    return pd.DataFrame([sample_booking_dict, low])
