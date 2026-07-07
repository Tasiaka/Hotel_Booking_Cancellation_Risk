from __future__ import annotations

from pathlib import Path
import socket

import pandas as pd

from .db import SessionLocal, init_db
from .ml import predict_dataframe
from .repository import create_scoring_batch, persist_predictions, update_scoring_job


def score_file_job(file_path: str, batch_name: str | None = None, scoring_job_id: int | None = None) -> dict:
    """RQ job executed by independently scalable model worker containers."""
    init_db()
    path = Path(file_path)
    worker_name = socket.gethostname()
    try:
        with SessionLocal() as session:
            if scoring_job_id is not None:
                update_scoring_job(session, scoring_job_id, status="scoring", worker_name=worker_name)
                session.commit()

        df = pd.read_csv(path)
        predictions = predict_dataframe(df, strict=False)
        with SessionLocal() as session:
            batch = create_scoring_batch(session, batch_name or path.stem, source="worker_csv", file_name=path.name)
            persist_predictions(session, batch, predictions)
            if scoring_job_id is not None:
                update_scoring_job(session, scoring_job_id, status="completed", batch_id=batch.id, worker_name=worker_name)
            session.commit()
            return {
                "batch_id": batch.id,
                "rows": batch.row_count,
                "high_risk_count": batch.high_risk_count,
                "total_expected_loss": batch.total_expected_loss,
                "scoring_job_id": scoring_job_id,
            }
    except Exception:
        if scoring_job_id is not None:
            with SessionLocal() as session:
                update_scoring_job(session, scoring_job_id, status="failed", worker_name=worker_name)
                session.commit()
        raise
