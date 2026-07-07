import pandas as pd

from hotel_risk.ml import _assign_hw4_booking_creation_split


def test_training_split_uses_booking_creation_date_when_split_missing():
    df = pd.DataFrame(
        {
            "arrival_date": ["2016-11-10", "2017-01-15", "2017-06-20"],
            "lead_time": [10, 20, 30],
            "arrival_date_year": [2016, 2017, 2017],
            "arrival_date_month": ["November", "January", "June"],
            "arrival_date_day_of_month": [10, 15, 20],
            "arrival_date_week_number": [45, 2, 25],
        }
    )

    out, strategy = _assign_hw4_booking_creation_split(df)

    assert strategy == "booking_creation_date_hw4"
    assert out["booking_creation_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2016-10-31",
        "2016-12-26",
        "2017-05-21",
    ]
    assert out["split"].tolist() == ["train", "valid", "test"]
