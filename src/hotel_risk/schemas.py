from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    model: str
    database: str


class DomainEntityResponse(BaseModel):
    name: str
    description: str


class DomainModelResponse(BaseModel):
    entities: list[DomainEntityResponse]
    relationships: list[str]
    storage: str
    scaling: str


class BookingPayload(BaseModel):
    hotel: str = "City Hotel"
    lead_time: int = 30
    arrival_date_year: int = 2017
    arrival_date_month: str = "August"
    arrival_date_day_of_month: int = 10
    arrival_date_week_number: int = 32
    stays_in_weekend_nights: int = 1
    stays_in_week_nights: int = 2
    adults: int = 2
    children: float = 0
    babies: int = 0
    meal: str = "BB"
    country: str = "PRT"
    market_segment: str = "Online TA"
    distribution_channel: str = "TA/TO"
    is_repeated_guest: int = 0
    previous_cancellations: int = 0
    previous_bookings_not_canceled: int = 0
    reserved_room_type: str = "A"
    deposit_type: str = "No Deposit"
    customer_type: str = "Transient"
    adr: float = 100.0
    required_car_parking_spaces: int = 0
    total_of_special_requests: int = 0
    booking_id: str | int | None = None


class ScoreRequest(BaseModel):
    bookings: list[dict[str, Any]] = Field(min_length=1)


class ValidateRequest(BaseModel):
    bookings: list[dict[str, Any]] = Field(min_length=1)


class ValidationResponse(BaseModel):
    ok: bool
    row_count: int
    column_count: int
    missing_required: list[str]
    leakage_removed: list[str]
    warnings: list[str]
    mapped_columns: dict[str, str]
    defaulted_columns: list[str]
    ignored_columns: list[str]
    preview: list[dict[str, Any]] = []


class PredictionResponse(BaseModel):
    booking_id: str
    hotel: str
    market_segment: str
    distribution_channel: str
    deposit_type: str
    adr: float
    total_nights: int
    booking_value: float
    risk_score: float
    cancellation_probability: float
    risk_category: str
    expected_loss: float
    business_priority_score: float
    deposit_loss_factor: float = 1.0
    actionable: bool = False
    financial_priority: str = ""
    recommended_action: str
    top_factors: list[str]


class ScoreResponse(BaseModel):
    summary: dict[str, Any]
    predictions: list[PredictionResponse]


class BatchResponse(BaseModel):
    id: int
    name: str
    source: str
    file_name: str | None
    status: str
    row_count: int
    high_risk_count: int
    total_booking_value: float
    total_expected_loss: float
    created_at: str | None = None
    updated_at: str | None = None
    error_message: str | None = None


class SchemaResponse(BaseModel):
    required_columns: list[str]
    optional_columns: list[str]
    leakage_columns: list[str]
    aliases: dict[str, list[str]]
    defaults: dict[str, Any]
    accepted_modes: list[str]


class ModelInfoResponse(BaseModel):
    model_path: str
    model_name: str
    metrics: dict[str, Any]
    feature_count: int
    numeric_features: list[str]
    categorical_features: list[str]
    risk_thresholds: dict[str, float]


class SimulationRequest(BaseModel):
    predictions: list[dict[str, Any]] = Field(min_length=1)
    top_share: float = Field(default=0.2, ge=0.01, le=1.0)
    intervention_success_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    cost_per_action: float = Field(default=150.0, ge=0.0)


class SimulationResponse(BaseModel):
    top_share: float
    intervention_success_rate: float
    cost_per_action: float
    result: dict[str, float]


class InsightResponse(BaseModel):
    summary: dict[str, Any]
    risk_distribution: list[dict[str, Any]]
    segment_expected_loss: list[dict[str, Any]]
    deposit_expected_loss: list[dict[str, Any]]
    top_recommendations: list[dict[str, Any]]


class OverviewResponse(BaseModel):
    batches_total: int
    bookings_scored_total: int
    high_risk_total: int
    expected_loss_total: float
    recent_batches: list[BatchResponse]
