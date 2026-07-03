from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi.testclient import TestClient

from hotel_risk.api.main import app


def test_schema_and_model_info_endpoints():
    client = TestClient(app)
    schema = client.get("/api/v1/schema")
    assert schema.status_code == 200
    body = schema.json()
    assert "lead_time" in body["required_columns"]
    assert "reservation_status" in body["leakage_columns"]
    assert "LeadTime" in body["aliases"]["lead_time"]

    info = client.get("/api/v1/model-info")
    assert info.status_code == 200
    assert info.json()["feature_count"] >= 10
    assert "numeric_features" in info.json()


def test_sample_csv_downloads_are_readable():
    client = TestClient(app)
    for kind in ["canonical", "flexible"]:
        r = client.get(f"/api/v1/sample-csv?format={kind}")
        assert r.status_code == 200
        assert "text/csv" in r.headers["content-type"]
        df = pd.read_csv(BytesIO(r.content))
        assert len(df) >= 1


def test_validate_and_score_csv_with_flexible_schema():
    client = TestClient(app)
    csv = b"id,hotel_type,LeadTime,arrival_date,no_of_week_nights,no_of_adults,avg_price_per_room,type_of_meal_plan,market_segment_type,booking_channel,room_type_reserved,no_of_special_requests\nF-1,City Hotel,90,2017-08-10,2,2,120,BB,Online TA,TA/TO,A,0\n"
    files = {"file": ("flex.csv", csv, "text/csv")}
    validate = client.post("/api/v1/validate-csv", files=files)
    assert validate.status_code == 200, validate.text
    assert validate.json()["row_count"] == 1
    assert validate.json()["defaulted_columns"]

    files = {"file": ("flex.csv", csv, "text/csv")}
    score = client.post("/api/v1/score-csv", files=files)
    assert score.status_code == 200, score.text
    assert score.json()["summary"]["rows"] == 1
    assert score.json()["predictions"][0]["booking_id"] == "F-1"


def test_simulate_endpoint(sample_booking_dict):
    client = TestClient(app)
    score = client.post("/api/v1/score", json={"bookings": [sample_booking_dict]})
    predictions = score.json()["predictions"]
    r = client.post("/api/v1/simulate", json={"predictions": predictions, "top_share": 1.0, "intervention_success_rate": 0.25, "cost_per_action": 0})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["selected_bookings"] == 1
    assert r.json()["result"]["gross_protected_revenue"] >= 0


def test_batch_insights_export_and_overview(two_bookings_df):
    client = TestClient(app)
    buf = BytesIO()
    two_bookings_df.to_csv(buf, index=False)
    buf.seek(0)
    upload = client.post("/api/v1/batches/upload", params={"name": "pro-test"}, files={"file": ("two.csv", buf.getvalue(), "text/csv")})
    assert upload.status_code == 200, upload.text
    batch_id = upload.json()["id"]

    insights = client.get(f"/api/v1/batches/{batch_id}/insights")
    assert insights.status_code == 200, insights.text
    assert insights.json()["summary"]["rows"] == 2
    assert isinstance(insights.json()["risk_distribution"], list)

    export = client.get(f"/api/v1/batches/{batch_id}/export")
    assert export.status_code == 200
    assert "text/csv" in export.headers["content-type"]
    assert "booking_id" in export.text

    overview = client.get("/api/v1/analytics/overview")
    assert overview.status_code == 200
    assert overview.json()["batches_total"] >= 1
