from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient

from hotel_risk.api.main import app


def test_score_endpoint_rejects_missing_required_columns():
    client = TestClient(app)
    response = client.post("/api/v1/score?strict=true", json={"bookings": [{"lead_time": 10}]})
    assert response.status_code == 422
    assert "Не хватает обязательных колонок" in response.text


def test_score_endpoint_rejects_empty_booking_list():
    client = TestClient(app)
    response = client.post("/api/v1/score", json={"bookings": []})
    assert response.status_code == 422


def test_upload_batch_rejects_non_csv_file():
    client = TestClient(app)
    response = client.post(
        "/api/v1/batches/upload",
        files={"file": ("bookings.txt", b"not,csv", "text/plain")},
    )
    assert response.status_code == 400
    assert "CSV" in response.text


def test_upload_batch_scores_csv_and_predictions_are_readable(sample_booking_dict):
    client = TestClient(app)
    header = list(sample_booking_dict.keys())
    row = [str(sample_booking_dict[col]) for col in header]
    csv_bytes = (",".join(header) + "\n" + ",".join(row) + "\n").encode("utf-8")
    response = client.post(
        "/api/v1/batches/upload?name=pytest-upload",
        files={"file": ("bookings.csv", BytesIO(csv_bytes), "text/csv")},
    )
    assert response.status_code == 200, response.text
    batch = response.json()
    assert batch["status"] == "completed"
    assert batch["row_count"] == 1

    detail = client.get(f"/api/v1/batches/{batch['id']}")
    assert detail.status_code == 200
    preds = client.get(f"/api/v1/batches/{batch['id']}/predictions")
    assert preds.status_code == 200
    assert len(preds.json()["predictions"]) == 1


def test_unknown_batch_returns_404():
    client = TestClient(app)
    assert client.get("/api/v1/batches/999999999").status_code == 404
    assert client.get("/api/v1/batches/999999999/predictions").status_code == 404


def test_score_endpoint_accepts_partial_csv_like_payload():
    client = TestClient(app)
    response = client.post("/api/v1/score", json={"bookings": [{"LeadTime": 95, "avg_price_per_room": 120, "no_of_adults": 2, "arrival_date": "2017-08-10"}]})
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert 0 <= data["predictions"][0]["risk_score"] <= 1
