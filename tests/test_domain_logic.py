from __future__ import annotations

import pytest

from hotel_risk.domain import (
    BookingAction,
    RiskCategory,
    categorize_risk,
    deposit_loss_factor,
    recommended_action,
)


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, RiskCategory.LOW),
        (0.2999, RiskCategory.LOW),
        (0.30, RiskCategory.MEDIUM),
        (0.5999, RiskCategory.MEDIUM),
        (0.60, RiskCategory.HIGH),
        (0.7999, RiskCategory.HIGH),
        (0.80, RiskCategory.CRITICAL),
        (1.0, RiskCategory.CRITICAL),
    ],
)
def test_categorize_risk_threshold_boundaries(score, expected):
    assert categorize_risk(score) == expected


@pytest.mark.parametrize(
    "deposit_type,expected",
    [
        ("No Deposit", 1.0),
        ("Refundable", 1.0),
        ("Non Refund", 0.0),
        ("Unknown", 1.0),
        (None, 1.0),
    ],
)
def test_deposit_loss_factor_defaults_to_conservative_full_loss(deposit_type, expected):
    assert deposit_loss_factor(deposit_type) == expected


def test_recommended_action_mapping():
    assert recommended_action(RiskCategory.LOW) == BookingAction.NO_ACTION.value
    assert recommended_action(RiskCategory.MEDIUM) == BookingAction.CONFIRMATION.value
    assert recommended_action(RiskCategory.HIGH) == BookingAction.DEPOSIT_REVIEW.value
    assert recommended_action(RiskCategory.CRITICAL) == BookingAction.MANUAL_REVIEW.value
