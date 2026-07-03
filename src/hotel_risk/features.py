from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .domain import LEAKAGE_COLUMNS, REQUIRED_COLUMNS

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
MONTHS_REV = {v: k for k, v in MONTHS.items()}
MONTH_ALIASES = {m.lower(): m for m in MONTHS}
MONTH_ALIASES.update({m[:3].lower(): m for m in MONTHS})

NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_year",
    "arrival_month_num",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "has_previous_cancellations",
    "adr",
    "booking_value",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "has_special_requests",
    "has_children",
    "is_long_lead_booking",
    "is_weekend_stay",
    "country_missing",
    "agent_missing",
    "has_company",
    "arrival_quarter",
    "arrival_dayofweek",
    "arrival_is_weekend",
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    "weekend_share",
    "adr_per_guest",
    "lead_time_x_value",
    "lead_time_x_requests",
    "lead_time_log",
    "adr_log",
    "value_per_night_log",
    "is_single_adult",
    "is_family",
    "has_parking_request",
    "no_special_no_parking",
    # Added by the train-only encoder inside the sklearn pipeline.
    # In prepare_features they are initialized with neutral values so the API schema stays stable.
    "agent_freq_log",
    "agent_target_mean_smooth",
    "company_freq_log",
    "company_target_mean_smooth",
    "country_target_mean_smooth",
    "market_segment_target_mean_smooth",
    "distribution_channel_target_mean_smooth",
    "deposit_type_target_mean_smooth",
]

CATEGORICAL_FEATURES = [
    "hotel",
    "arrival_date_month",
    "arrival_season",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

# Raw columns needed only by the train-only target/frequency encoder. They are not one-hot encoded.
TARGET_ENCODING_INPUT_FEATURES = ["agent", "company_raw"]
TARGET_ENCODING_OUTPUT_FEATURES = [
    "agent_freq_log",
    "agent_target_mean_smooth",
    "company_freq_log",
    "company_target_mean_smooth",
    "country_target_mean_smooth",
    "market_segment_target_mean_smooth",
    "distribution_channel_target_mean_smooth",
    "deposit_type_target_mean_smooth",
]

FEATURE_COLUMNS = list(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_ENCODING_INPUT_FEATURES))

# Canonical column -> common aliases from Hotel Booking Demand, Hotel Reservations and user CSV exports.
COLUMN_ALIASES: dict[str, list[str]] = {
    "booking_id": ["booking_id", "booking id", "Booking_ID", "reservation_id", "reservation id", "id", "ID", "reference", "booking_reference"],
    "hotel": ["hotel", "hotel_type", "hotel type", "property", "property_type", "property type"],
    "lead_time": ["lead_time", "lead time", "LeadTime", "lead", "days_before_arrival", "days before arrival", "advance_days", "booking_window"],
    "arrival_date": ["arrival_date", "arrival date", "arrival", "checkin_date", "check-in date", "check_in_date", "date_of_arrival", "date of arrival"],
    "arrival_date_year": ["arrival_date_year", "arrival year", "arrival_year", "year", "checkin_year"],
    "arrival_date_month": ["arrival_date_month", "arrival month", "arrival_month", "month", "checkin_month"],
    "arrival_date_day_of_month": ["arrival_date_day_of_month", "arrival day", "arrival_day", "arrival_date", "day", "day_of_month", "checkin_day"],
    "arrival_date_week_number": ["arrival_date_week_number", "arrival week", "arrival_week", "week", "week_number", "week no"],
    "stays_in_weekend_nights": ["stays_in_weekend_nights", "weekend_nights", "no_of_weekend_nights", "weekend nights"],
    "stays_in_week_nights": ["stays_in_week_nights", "week_nights", "weekday_nights", "no_of_week_nights", "week nights"],
    "adults": ["adults", "no_of_adults", "adult", "adults_count"],
    "children": ["children", "no_of_children", "kids", "child"],
    "babies": ["babies", "infants", "baby"],
    "meal": ["meal", "meal_plan", "meal plan", "type_of_meal_plan", "meal_type"],
    "country": ["country", "guest_country", "customer_country", "nationality"],
    "market_segment": ["market_segment", "market segment", "market_segment_type", "segment", "sales_segment"],
    "distribution_channel": ["distribution_channel", "distribution channel", "channel", "booking_channel", "sales_channel"],
    "is_repeated_guest": ["is_repeated_guest", "repeated_guest", "repeat_guest", "repeated guest", "is_repeat"],
    "previous_cancellations": ["previous_cancellations", "previous cancellations", "no_of_previous_cancellations", "prev_cancellations"],
    "previous_bookings_not_canceled": ["previous_bookings_not_canceled", "previous bookings not canceled", "no_of_previous_bookings_not_canceled", "prev_not_canceled"],
    "reserved_room_type": ["reserved_room_type", "reserved room type", "room_type_reserved", "room_type", "room type", "reserved_room"],
    "deposit_type": ["deposit_type", "deposit type", "deposit", "deposit_policy"],
    "customer_type": ["customer_type", "customer type", "customer", "guest_type"],
    "adr": ["adr", "avg_price_per_room", "average_daily_rate", "average daily rate", "rate", "price", "room_price", "avg_price"],
    "required_car_parking_spaces": ["required_car_parking_spaces", "required_car_parking_space", "parking", "parking_spaces", "car_parking_spaces"],
    "total_of_special_requests": ["total_of_special_requests", "no_of_special_requests", "special_requests", "special requests", "requests"],
    "agent": ["agent", "agent_id", "agent id", "agency_id", "agency", "travel_agent"],
    "company_raw": ["company", "company_id", "company id", "corporate_id", "corp_id"],
    "has_company": ["has_company", "company_present", "corporate_booking"],
}

# Defaults are intentionally conservative. They let the MVP score partially wrong CSVs, but the
# response/report should still tell the user that the result is best-effort, not production-grade.
DEFAULT_VALUES: dict[str, object] = {
    "hotel": "City Hotel",
    "lead_time": 30,
    "arrival_date_year": 2017,
    "arrival_date_month": "August",
    "arrival_date_day_of_month": 1,
    "arrival_date_week_number": 32,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 1,
    "adults": 1,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "Unknown",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "A",
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "adr": 100.0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0,
    "agent": "Unknown",
    "company_raw": "Unknown",
    "has_company": 0,
}



@dataclass(frozen=True)
class ValidationReport:
    row_count: int
    column_count: int
    missing_required: list[str]
    leakage_removed: list[str]
    warnings: list[str]
    mapped_columns: dict[str, str] | None = None
    defaulted_columns: list[str] | None = None
    ignored_columns: list[str] | None = None

    @property
    def ok(self) -> bool:
        return not self.missing_required


def _norm_name(name: object) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _month_to_name(value: object) -> object:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        if 1 <= int(value) <= 12:
            return MONTHS_REV[int(value)]
    text = str(value).strip()
    if text.isdigit() and 1 <= int(text) <= 12:
        return MONTHS_REV[int(text)]
    return MONTH_ALIASES.get(text.lower(), text)


def season(month: int | float | None) -> str:
    if pd.isna(month):
        return "unknown"
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def normalize_input_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], list[str], list[str]]:
    """Map user CSV columns to the canonical Hotel Booking schema and fill safe defaults.

    This is intentionally permissive for demo/MVP use. It does not make arbitrary data reliable;
    it only extracts recognizable booking fields and fills missing features with conservative
    defaults so the service can return a best-effort score instead of crashing.
    """
    out = df.copy()
    original_columns = list(out.columns)
    norm_to_original: dict[str, str] = {}
    for col in original_columns:
        norm_to_original.setdefault(_norm_name(col), col)

    mapped: dict[str, str] = {}
    renamed: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            mapped[canonical] = canonical
            continue
        for alias in aliases:
            source = norm_to_original.get(_norm_name(alias))
            if source is not None:
                renamed[source] = canonical
                mapped[canonical] = source
                break
    if renamed:
        out = out.rename(columns=renamed)

    # If a single arrival_date exists, derive year/month/day/week from it.
    if "arrival_date" in out.columns:
        arrival = pd.to_datetime(out["arrival_date"], errors="coerce")
        if "arrival_date_year" not in out.columns:
            out["arrival_date_year"] = arrival.dt.year
        if "arrival_date_month" not in out.columns:
            out["arrival_date_month"] = arrival.dt.month.map(MONTHS_REV)
        if "arrival_date_day_of_month" not in out.columns:
            out["arrival_date_day_of_month"] = arrival.dt.day
        if "arrival_date_week_number" not in out.columns:
            out["arrival_date_week_number"] = arrival.dt.isocalendar().week.astype(float)

    if "arrival_date_month" in out.columns:
        out["arrival_date_month"] = out["arrival_date_month"].map(_month_to_name)

    defaulted: list[str] = []
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = DEFAULT_VALUES.get(col, np.nan)
            defaulted.append(col)

    kept_special = set(REQUIRED_COLUMNS) | {"booking_id", "arrival_date", "arrival_month_num", "is_canceled", "agent", "company_raw", "has_company"}
    ignored = [c for c in original_columns if renamed.get(c, c) not in kept_special and c not in LEAKAGE_COLUMNS]
    return out, mapped, defaulted, ignored


def validate_input(df: pd.DataFrame, allow_defaults: bool = False) -> ValidationReport:
    normalized, mapped, defaulted, ignored = normalize_input_schema(df)
    # Before defaults, the missing canonical columns are exactly the columns that had to be defaulted.
    missing_required = [] if allow_defaults else list(defaulted)
    leakage_removed = [c for c in LEAKAGE_COLUMNS if c in normalized.columns]
    warnings: list[str] = []
    if "is_canceled" in normalized.columns:
        warnings.append("Целевая переменная is_canceled найдена и будет исключена из inference-признаков.")
    if leakage_removed:
        warnings.append("Post-factum признаки обнаружены и будут удалены: " + ", ".join(leakage_removed))
    if defaulted:
        warnings.append("Часть обязательных признаков отсутствовала и была заполнена дефолтами: " + ", ".join(defaulted))
    if ignored:
        warnings.append("Лишние колонки не используются моделью: " + ", ".join(ignored[:20]) + ("..." if len(ignored) > 20 else ""))
    return ValidationReport(
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        missing_required=missing_required,
        leakage_removed=leakage_removed,
        warnings=warnings,
        mapped_columns=mapped,
        defaulted_columns=defaulted,
        ignored_columns=ignored,
    )


def prepare_features(df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, ValidationReport]:
    out, mapped, defaulted, ignored = normalize_input_schema(df)
    report = validate_input(df, allow_defaults=not strict)
    if strict and defaulted:
        raise ValueError("Не хватает обязательных колонок: " + ", ".join(defaulted))
    if strict and not report.ok:
        raise ValueError("Не хватает обязательных колонок: " + ", ".join(report.missing_required))

    out = out.drop(columns=[c for c in LEAKAGE_COLUMNS if c in out.columns], errors="ignore")
    out = _ensure_columns(out, REQUIRED_COLUMNS + ["country", "arrival_date_week_number", "booking_id", "agent", "company_raw", "has_company"])

    if "booking_id" not in out.columns or out["booking_id"].isna().all():
        out["booking_id"] = np.arange(1, len(out) + 1)

    out["children"] = pd.to_numeric(out["children"], errors="coerce").fillna(0)
    out["babies"] = pd.to_numeric(out["babies"], errors="coerce").fillna(0)
    out["adults"] = pd.to_numeric(out["adults"], errors="coerce").fillna(0)
    out["lead_time"] = pd.to_numeric(out["lead_time"], errors="coerce").fillna(0).clip(lower=0)
    out["adr"] = pd.to_numeric(out["adr"], errors="coerce").fillna(0).clip(lower=0)
    out["stays_in_weekend_nights"] = pd.to_numeric(out["stays_in_weekend_nights"], errors="coerce").fillna(0).clip(lower=0)
    out["stays_in_week_nights"] = pd.to_numeric(out["stays_in_week_nights"], errors="coerce").fillna(0).clip(lower=0)
    out["required_car_parking_spaces"] = pd.to_numeric(out["required_car_parking_spaces"], errors="coerce").fillna(0).clip(lower=0)
    out["total_of_special_requests"] = pd.to_numeric(out["total_of_special_requests"], errors="coerce").fillna(0).clip(lower=0)
    out["previous_cancellations"] = pd.to_numeric(out["previous_cancellations"], errors="coerce").fillna(0).clip(lower=0)
    out["previous_bookings_not_canceled"] = pd.to_numeric(out["previous_bookings_not_canceled"], errors="coerce").fillna(0).clip(lower=0)
    out["is_repeated_guest"] = pd.to_numeric(out["is_repeated_guest"], errors="coerce").fillna(0).clip(0, 1)

    out["country_missing"] = out["country"].isna().astype(int)
    out["country"] = out["country"].astype("string").fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    out["agent_missing"] = out["agent"].isna().astype(int)
    out["agent"] = out["agent"].astype("string").fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    out["company_raw"] = out["company_raw"].astype("string").fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    if "has_company" not in out.columns or out["has_company"].isna().all():
        out["has_company"] = (out["company_raw"].astype(str).str.lower().ne("unknown")).astype(int)
    else:
        out["has_company"] = pd.to_numeric(out["has_company"], errors="coerce").fillna(0).clip(0, 1)

    if "arrival_month_num" not in out.columns or out["arrival_month_num"].isna().all():
        out["arrival_month_num"] = out["arrival_date_month"].map(MONTHS)
    out["arrival_month_num"] = pd.to_numeric(out["arrival_month_num"], errors="coerce").fillna(1).clip(1, 12)
    out["arrival_date_month"] = out["arrival_date_month"].fillna(out["arrival_month_num"].astype(int).map(MONTHS_REV)).fillna("Unknown")

    if "arrival_date" in out.columns:
        arrival_date = pd.to_datetime(out["arrival_date"], errors="coerce")
    else:
        arrival_date = pd.to_datetime(
            dict(
                year=pd.to_numeric(out["arrival_date_year"], errors="coerce").fillna(2017).astype(int),
                month=out["arrival_month_num"].fillna(1).astype(int),
                day=pd.to_numeric(out["arrival_date_day_of_month"], errors="coerce").fillna(1).clip(1, 28).astype(int),
            ),
            errors="coerce",
        )
    out["arrival_date"] = arrival_date
    out["arrival_date_year"] = pd.to_numeric(out["arrival_date_year"], errors="coerce").fillna(out["arrival_date"].dt.year).fillna(2017)
    out["arrival_date_week_number"] = pd.to_numeric(out["arrival_date_week_number"], errors="coerce").fillna(out["arrival_date"].dt.isocalendar().week.astype(float)).fillna(1)
    out["arrival_date_day_of_month"] = pd.to_numeric(out["arrival_date_day_of_month"], errors="coerce").fillna(out["arrival_date"].dt.day).fillna(1)
    out["arrival_season"] = out["arrival_month_num"].map(season)

    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out.loc[out["total_nights"] <= 0, "total_nights"] = 1
    out["total_guests"] = out["adults"] + out["children"] + out["babies"]
    out.loc[out["total_guests"] <= 0, "total_guests"] = 1
    out["booking_value"] = out["adr"] * out["total_nights"]

    out["has_children"] = ((out["children"] + out["babies"]) > 0).astype(int)
    out["has_previous_cancellations"] = (out["previous_cancellations"] > 0).astype(int)
    out["has_special_requests"] = (out["total_of_special_requests"] > 0).astype(int)
    out["is_long_lead_booking"] = (out["lead_time"] > 90).astype(int)
    out["is_weekend_stay"] = (out["stays_in_weekend_nights"] > 0).astype(int)
    out["arrival_quarter"] = np.ceil(out["arrival_month_num"] / 3).astype(int)
    out["arrival_dayofweek"] = out["arrival_date"].dt.dayofweek.fillna(0).astype(int)
    out["arrival_is_weekend"] = out["arrival_dayofweek"].isin([5, 6]).astype(int)
    out["month_sin"] = np.sin(2 * math.pi * out["arrival_month_num"] / 12)
    out["month_cos"] = np.cos(2 * math.pi * out["arrival_month_num"] / 12)
    out["week_sin"] = np.sin(2 * math.pi * out["arrival_date_week_number"] / 53)
    out["week_cos"] = np.cos(2 * math.pi * out["arrival_date_week_number"] / 53)
    out["weekend_share"] = out["stays_in_weekend_nights"] / out["total_nights"].clip(lower=1)
    out["adr_per_guest"] = out["adr"] / out["total_guests"].clip(lower=1)
    out["lead_time_x_value"] = out["lead_time"] * np.log1p(out["booking_value"].clip(lower=0))
    out["lead_time_x_requests"] = out["lead_time"] * (1 + out["total_of_special_requests"])
    out["lead_time_log"] = np.log1p(out["lead_time"].clip(lower=0))
    out["adr_log"] = np.log1p(out["adr"].clip(lower=0))
    out["value_per_night_log"] = np.log1p((out["booking_value"] / out["total_nights"].clip(lower=1)).clip(lower=0))
    out["is_single_adult"] = ((out["adults"] == 1) & (out["children"] == 0) & (out["babies"] == 0)).astype(int)
    out["is_family"] = ((out["adults"] >= 1) & ((out["children"] + out["babies"]) > 0)).astype(int)
    out["has_parking_request"] = (out["required_car_parking_spaces"] > 0).astype(int)
    out["no_special_no_parking"] = ((out["total_of_special_requests"] == 0) & (out["required_car_parking_spaces"] == 0)).astype(int)

    for col in TARGET_ENCODING_OUTPUT_FEATURES:
        if col not in out.columns:
            out[col] = 0.0

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].astype("string").fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"})

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    # Return an updated report with actual mapping/default metadata.
    final_report = ValidationReport(
        row_count=report.row_count,
        column_count=report.column_count,
        missing_required=[] if not strict else report.missing_required,
        leakage_removed=report.leakage_removed,
        warnings=report.warnings,
        mapped_columns=mapped,
        defaulted_columns=defaulted,
        ignored_columns=ignored,
    )
    return out, final_report


def model_matrix(df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
    prepared, report = prepare_features(df, strict=strict)
    return prepared[FEATURE_COLUMNS].copy(), prepared, report
