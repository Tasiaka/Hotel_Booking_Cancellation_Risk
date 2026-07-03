from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotel_risk.domain import LEAKAGE_COLUMNS
from hotel_risk.features import FEATURE_COLUMNS, NUMERIC_FEATURES, ValidationReport, model_matrix, prepare_features, season, validate_input


@pytest.mark.parametrize(
    "month,expected",
    [(1, "winter"), (4, "spring"), (7, "summer"), (10, "autumn"), (np.nan, "unknown")],
)
def test_season_mapping(month, expected):
    assert season(month) == expected


def test_validate_input_reports_missing_required_columns():
    report = validate_input(pd.DataFrame([{"lead_time": 10}]))
    assert isinstance(report, ValidationReport)
    assert not report.ok
    assert "hotel" in report.missing_required


def test_prepare_features_strict_raises_on_missing_required_columns():
    with pytest.raises(ValueError, match="Не хватает обязательных колонок"):
        prepare_features(pd.DataFrame([{"lead_time": 10}]), strict=True)


def test_prepare_features_removes_all_leakage_columns_and_target_warning(sample_booking_dict):
    row = dict(sample_booking_dict)
    row.update({col: "leak" for col in LEAKAGE_COLUMNS})
    row["is_canceled"] = 1
    prepared, report = prepare_features(pd.DataFrame([row]))
    assert set(report.leakage_removed) == set(LEAKAGE_COLUMNS)
    assert any("is_canceled" in warning for warning in report.warnings)
    assert not any(col in prepared.columns for col in LEAKAGE_COLUMNS)


def test_prepare_features_engineers_core_numeric_values(sample_booking_df):
    prepared, report = prepare_features(sample_booking_df)
    assert report.ok
    assert prepared.loc[0, "total_nights"] == 3
    assert prepared.loc[0, "total_guests"] == 2
    assert prepared.loc[0, "booking_value"] == 360.0
    assert prepared.loc[0, "has_previous_cancellations"] == 1
    assert prepared.loc[0, "has_special_requests"] == 0
    assert prepared.loc[0, "is_long_lead_booking"] == 1
    assert prepared.loc[0, "arrival_season"] == "summer"
    assert all(col in prepared.columns for col in FEATURE_COLUMNS)


def test_prepare_features_clips_invalid_numeric_values(sample_booking_dict):
    row = dict(sample_booking_dict)
    row.update({"lead_time": -10, "adr": -50, "stays_in_weekend_nights": -1, "stays_in_week_nights": -2, "adults": 0})
    prepared, _ = prepare_features(pd.DataFrame([row]))
    assert prepared.loc[0, "lead_time"] == 0
    assert prepared.loc[0, "adr"] == 0
    assert prepared.loc[0, "total_nights"] == 1
    assert prepared.loc[0, "total_guests"] == 1
    assert prepared.loc[0, "booking_value"] == 0


def test_model_matrix_returns_exact_feature_set(sample_booking_df):
    x, prepared, report = model_matrix(sample_booking_df)
    assert report.ok
    assert list(x.columns) == FEATURE_COLUMNS
    assert len(x) == len(prepared) == 1
    assert set(NUMERIC_FEATURES).issubset(x.columns)


def test_prepare_features_lenient_accepts_wrong_schema():
    from hotel_risk.features import prepare_features
    df = pd.DataFrame([{
        "LeadTime": 120,
        "avg_price_per_room": 150,
        "no_of_week_nights": 2,
        "no_of_adults": 2,
        "arrival_date": "2017-08-10",
        "type_of_meal_plan": "BB",
    }])
    prepared, report = prepare_features(df, strict=False)
    assert prepared.loc[0, "lead_time"] == 120
    assert prepared.loc[0, "adr"] == 150
    assert prepared.loc[0, "arrival_date_month"] == "August"
    assert report.defaulted_columns
