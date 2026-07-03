from __future__ import annotations

import pandas as pd

from hotel_risk.worker_jobs import score_file_job


def test_score_file_job_processes_csv_and_returns_batch_summary(tmp_path, sample_booking_dict):
    path = tmp_path / "worker_bookings.csv"
    pd.DataFrame([sample_booking_dict]).to_csv(path, index=False)
    result = score_file_job(str(path), batch_name="worker-pytest")
    assert result["batch_id"] >= 1
    assert result["rows"] == 1
    assert "high_risk_count" in result
    assert "total_expected_loss" in result
