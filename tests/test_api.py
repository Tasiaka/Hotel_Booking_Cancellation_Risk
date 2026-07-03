from __future__ import annotations

from fastapi.testclient import TestClient

from hotel_risk.api.main import app


def test_health_and_domain_model():
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["status"] == "ok"
    d = client.get("/api/v1/domain-model")
    assert d.status_code == 200
    names = [e["name"] for e in d.json()["entities"]]
    assert "Booking" in names
    assert "Prediction" in names


def test_score_endpoint():
    client = TestClient(app)
    payload = {
        "bookings": [
            {
                "booking_id": "B-42",
                "hotel": "City Hotel",
                "lead_time": 120,
                "arrival_date_year": 2017,
                "arrival_date_month": "August",
                "arrival_date_week_number": 32,
                "arrival_date_day_of_month": 10,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 2,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "deposit_type": "No Deposit",
                "customer_type": "Transient",
                "adr": 100.0,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 0,
            }
        ]
    }
    r = client.post("/api/v1/score", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["rows"] == 1
    assert body["predictions"][0]["booking_id"] == "B-42"
    assert 0 <= body["predictions"][0]["risk_score"] <= 1
