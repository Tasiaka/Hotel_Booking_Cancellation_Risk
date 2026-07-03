from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class RiskCategory(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class BatchStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    SCORING = "scoring"
    COMPLETED = "completed"
    FAILED = "failed"


class BookingAction(StrEnum):
    NO_ACTION = "Без дополнительных действий"
    LOW_FINANCIAL_PRIORITY = "Не требует ручной проверки"
    CONFIRMATION = "Отправить напоминание и запрос подтверждения"
    DEPOSIT_REVIEW = "Проверить условия бронирования и запросить подтверждение"
    MANUAL_REVIEW = "Передать на ручную проверку"


RISK_THRESHOLDS: Final[dict[RiskCategory, float]] = {
    RiskCategory.CRITICAL: 0.80,
    RiskCategory.HIGH: 0.60,
    RiskCategory.MEDIUM: 0.30,
    RiskCategory.LOW: 0.0,
}

DEPOSIT_LOSS_FACTOR: Final[dict[str, float]] = {
    # For monetary prioritization the model estimates cancellation probability.
    # The tariff layer only decides whether cancellation can create a revenue loss.
    # Refundable and No Deposit are both treated as fully exposed: the model's
    # probability is not manually discounted.
    "No Deposit": 1.00,
    "Refundable": 1.00,
    # Non Refund is non-refundable revenue for the hotel, so cancellation does
    # not create lost revenue in the manager's financial queue.
    "Non Refund": 0.00,
}

MIN_ACTION_EXPECTED_LOSS: Final[float] = 100.0

LEAKAGE_COLUMNS: Final[list[str]] = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "days_in_waiting_list",
]

REQUIRED_COLUMNS: Final[list[str]] = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]


@dataclass(frozen=True)
class DomainEntity:
    name: str
    description: str


DOMAIN_MODEL: Final[list[DomainEntity]] = [
    DomainEntity("HotelProperty", "Объект размещения, валюта, регион и настройки пилота."),
    DomainEntity("ScoringBatch", "Пакет CSV-загрузки: источник, статус обработки, агрегированные KPI."),
    DomainEntity("Booking", "Бронирование с параметрами гостя, канала, дат, депозита и стоимости."),
    DomainEntity("Prediction", "Результат модели: raw risk_score, calibrated cancellation_probability, risk_category, expected_loss, recommended_action."),
    DomainEntity("ScoringJob", "Асинхронная задача скоринга, выполняемая масштабируемыми model workers."),
    DomainEntity("AuditLog", "Журнал действий: загрузка файла, валидация, скоринг, экспорт."),
]


def categorize_risk(score: float) -> RiskCategory:
    if score >= RISK_THRESHOLDS[RiskCategory.CRITICAL]:
        return RiskCategory.CRITICAL
    if score >= RISK_THRESHOLDS[RiskCategory.HIGH]:
        return RiskCategory.HIGH
    if score >= RISK_THRESHOLDS[RiskCategory.MEDIUM]:
        return RiskCategory.MEDIUM
    return RiskCategory.LOW


def recommended_action(category: RiskCategory, expected_loss: float | None = None) -> str:
    # If the probability is high but the expected monetary loss is absent or tiny,
    # the booking is not a manager priority. When expected_loss is omitted, return
    # the pure category mapping for documentation/tests.
    if expected_loss is not None and expected_loss < MIN_ACTION_EXPECTED_LOSS and category in {RiskCategory.HIGH, RiskCategory.CRITICAL}:
        return BookingAction.LOW_FINANCIAL_PRIORITY.value
    if category == RiskCategory.CRITICAL:
        return BookingAction.MANUAL_REVIEW.value
    if category == RiskCategory.HIGH:
        return BookingAction.DEPOSIT_REVIEW.value
    if category == RiskCategory.MEDIUM:
        return BookingAction.CONFIRMATION.value
    return BookingAction.NO_ACTION.value


def deposit_loss_factor(deposit_type: str | None) -> float:
    return DEPOSIT_LOSS_FACTOR.get(str(deposit_type or "No Deposit"), 1.0)
