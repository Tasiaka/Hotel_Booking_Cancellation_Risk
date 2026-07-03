from __future__ import annotations

from pathlib import Path

import pandas as pd

from .db import SessionLocal, init_db
from .ml import predict_dataframe
from .repository import create_scoring_batch, persist_predictions


def score_file_job(file_path: str, batch_name: str | None = None) -> dict:
    """RQ job executed by independently scalable model worker containers."""
    init_db()
    path = Path(file_path)
    df = pd.read_csv(path)
    predictions = predict_dataframe(df, strict=False)
    with SessionLocal() as session:
        batch = create_scoring_batch(session, batch_name or path.stem, source="worker_csv", file_name=path.name)
        persist_predictions(session, batch, predictions)
        session.commit()
        return {
            "batch_id": batch.id,
            "rows": batch.row_count,
            "high_risk_count": batch.high_risk_count,
            "total_expected_loss": batch.total_expected_loss,
        }
