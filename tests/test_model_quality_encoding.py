from __future__ import annotations

import numpy as np
import pandas as pd

from hotel_risk.features import FEATURE_COLUMNS, TARGET_ENCODING_OUTPUT_FEATURES, prepare_features
from hotel_risk.ml import TrainOnlyTargetFrequencyEncoder


def test_train_only_encoder_adds_expected_features_without_leaking_unknown_categories():
    x = pd.DataFrame(
        {
            "agent": ["A", "A", "B", "C"],
            "company_raw": ["Unknown", "10", "10", "20"],
            "country": ["PRT", "PRT", "GBR", "ESP"],
            "market_segment": ["Online TA", "Online TA", "Direct", "Groups"],
            "distribution_channel": ["TA/TO", "TA/TO", "Direct", "TA/TO"],
            "deposit_type": ["No Deposit", "No Deposit", "Refundable", "Non Refund"],
        }
    )
    y = np.array([1, 1, 0, 0])
    enc = TrainOnlyTargetFrequencyEncoder(smoothing=10.0).fit(x, y)
    transformed = enc.transform(pd.DataFrame({"agent": ["A", "new"], "company_raw": ["10", "new"]}))
    for col in TARGET_ENCODING_OUTPUT_FEATURES:
        assert col in transformed.columns
    assert transformed.loc[0, "agent_freq_log"] > transformed.loc[1, "agent_freq_log"]
    assert 0.0 <= transformed.loc[1, "agent_target_mean_smooth"] <= 1.0


def test_prepared_feature_schema_contains_quality_encoding_columns(two_bookings_df):
    prepared, _ = prepare_features(two_bookings_df, strict=False)
    for col in TARGET_ENCODING_OUTPUT_FEATURES:
        assert col in prepared.columns
        assert col in FEATURE_COLUMNS
