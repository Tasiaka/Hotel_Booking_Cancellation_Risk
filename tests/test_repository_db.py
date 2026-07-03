from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from hotel_risk.business import enrich_predictions
from hotel_risk.db import AuditLogORM, Base, batch_to_dict, make_engine
from hotel_risk.features import prepare_features
from hotel_risk.repository import create_scoring_batch, get_predictions, list_batches, persist_predictions


def test_repository_persists_batch_bookings_predictions_and_audit(two_bookings_df):
    engine = make_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    prepared, _ = prepare_features(two_bookings_df)
    predictions = enrich_predictions(prepared, [0.85, 0.20])

    with Session() as session:
        batch = create_scoring_batch(session, "pytest-batch", source="unit")
        persist_predictions(session, batch, predictions)
        session.commit()
        batch_id = batch.id

    with Session() as session:
        batches = list_batches(session, limit=10)
        assert len(batches) == 1
        assert batches[0].row_count == 2
        assert batch_to_dict(batches[0])["status"] == "completed"
        preds = get_predictions(session, batch_id=batch_id, limit=10)
        assert len(preds) == 2
        assert preds[0]["business_priority_score"] >= preds[1]["business_priority_score"]
        critical_only = get_predictions(session, batch_id=batch_id, limit=10, category="Critical")
        assert len(critical_only) == 1
        assert session.query(AuditLogORM).count() >= 2
