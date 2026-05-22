"""
Prepare datasets for Hotel Booking Cancellation Risk, homework 4.
Run from repository root:
    python src/prepare_dataset.py --main data/raw/hotel_bookings.csv --additional "data/raw/Hotel Reservations.csv"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MONTHS = {m: i for i, m in enumerate([
    'January','February','March','April','May','June','July','August','September','October','November','December'
], 1)}
MONTHS_REV = {v: k for k, v in MONTHS.items()}


def season(month: int) -> str:
    if month in [12, 1, 2]:
        return 'winter'
    if month in [3, 4, 5]:
        return 'spring'
    if month in [6, 7, 8]:
        return 'summer'
    return 'autumn'


def prepare_main(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['booking_id'] = np.arange(1, len(df) + 1)
    df['children'] = df['children'].fillna(0)
    df['arrival_month_num'] = df['arrival_date_month'].map(MONTHS)
    df['arrival_date'] = pd.to_datetime(dict(
        year=df.arrival_date_year,
        month=df.arrival_month_num,
        day=df.arrival_date_day_of_month,
    ), errors='coerce')
    df['booking_creation_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='D')
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['has_children'] = ((df['children'] + df['babies']) > 0).astype(int)
    df['has_previous_cancellations'] = (df['previous_cancellations'] > 0).astype(int)
    df['has_special_requests'] = (df['total_of_special_requests'] > 0).astype(int)
    df['is_long_lead_booking'] = (df['lead_time'] > 90).astype(int)
    df['is_weekend_stay'] = (df['stays_in_weekend_nights'] > 0).astype(int)
    df['country_missing'] = df['country'].isna().astype(int)
    df['country'] = df['country'].fillna('Unknown')
    df['agent_missing'] = df['agent'].isna().astype(int)
    df['agent'] = df['agent'].fillna(0).astype(int).astype(str)
    df['has_company'] = df['company'].notna().astype(int)
    df['booking_value'] = df['adr'] * df['total_nights']
    df['arrival_season'] = df['arrival_month_num'].map(season)
    df['source_dataset'] = 'hotel_booking_demand'

    df = df[(df.total_guests > 0) & (df.total_nights > 0) & (df.adr >= 0) & df.arrival_date.notna()].copy()
    df['split'] = np.select(
        [df.booking_creation_date <= pd.Timestamp('2016-10-31'), df.booking_creation_date <= pd.Timestamp('2017-03-31')],
        ['train', 'valid'],
        default='test',
    )
    allowed_cols = [
        'source_dataset','booking_id','split','hotel','is_canceled','lead_time','arrival_date','booking_creation_date',
        'arrival_date_year','arrival_month_num','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','arrival_season',
        'stays_in_weekend_nights','stays_in_week_nights','total_nights','adults','children','babies','total_guests','has_children',
        'meal','country','country_missing','market_segment','distribution_channel','is_repeated_guest','previous_cancellations',
        'previous_bookings_not_canceled','has_previous_cancellations','reserved_room_type','deposit_type','agent','agent_missing','has_company',
        'customer_type','adr','booking_value','required_car_parking_spaces','total_of_special_requests','has_special_requests','is_long_lead_booking','is_weekend_stay'
    ]
    return df[allowed_cols]


def prepare_additional(path: str | Path, columns_like: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['is_canceled'] = (df['booking_status'] == 'Canceled').astype(int)
    df['arrival_date_dt'] = pd.to_datetime(dict(year=df.arrival_year, month=df.arrival_month, day=df.arrival_date), errors='coerce')
    df['booking_creation_date'] = df['arrival_date_dt'] - pd.to_timedelta(df['lead_time'], unit='D')
    df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
    df['total_guests'] = df['no_of_adults'] + df['no_of_children']
    df['has_children'] = (df['no_of_children'] > 0).astype(int)
    df['has_previous_cancellations'] = (df['no_of_previous_cancellations'] > 0).astype(int)
    df['has_special_requests'] = (df['no_of_special_requests'] > 0).astype(int)
    df['booking_value'] = df['avg_price_per_room'] * df['total_nights']
    out = pd.DataFrame({
        'source_dataset': 'hotel_reservations_classification',
        'booking_id': df['Booking_ID'],
        'split': 'external',
        'hotel': 'Unknown',
        'is_canceled': df['is_canceled'],
        'lead_time': df['lead_time'],
        'arrival_date': df['arrival_date_dt'],
        'booking_creation_date': df['booking_creation_date'],
        'arrival_date_year': df['arrival_year'],
        'arrival_month_num': df['arrival_month'],
        'arrival_date_month': df['arrival_month'].map(MONTHS_REV),
        'arrival_date_week_number': np.nan,
        'arrival_date_day_of_month': df['arrival_date'],
        'arrival_season': df['arrival_month'].map(season),
        'stays_in_weekend_nights': df['no_of_weekend_nights'],
        'stays_in_week_nights': df['no_of_week_nights'],
        'total_nights': df['total_nights'],
        'adults': df['no_of_adults'],
        'children': df['no_of_children'],
        'babies': 0,
        'total_guests': df['total_guests'],
        'has_children': df['has_children'],
        'meal': df['type_of_meal_plan'],
        'country': 'Unknown',
        'country_missing': 1,
        'market_segment': df['market_segment_type'],
        'distribution_channel': 'Unknown',
        'is_repeated_guest': df['repeated_guest'],
        'previous_cancellations': df['no_of_previous_cancellations'],
        'previous_bookings_not_canceled': df['no_of_previous_bookings_not_canceled'],
        'has_previous_cancellations': df['has_previous_cancellations'],
        'reserved_room_type': df['room_type_reserved'],
        'deposit_type': 'Unknown',
        'agent': 'Unknown',
        'agent_missing': 1,
        'has_company': 0,
        'customer_type': 'Unknown',
        'adr': df['avg_price_per_room'],
        'booking_value': df['booking_value'],
        'required_car_parking_spaces': df['required_car_parking_space'],
        'total_of_special_requests': df['no_of_special_requests'],
        'has_special_requests': df['has_special_requests'],
        'is_long_lead_booking': (df['lead_time'] > 90).astype(int),
        'is_weekend_stay': (df['no_of_weekend_nights'] > 0).astype(int),
    })
    out = out[(out.total_guests > 0) & (out.total_nights > 0) & (out.adr >= 0) & out.arrival_date.notna()].copy()
    return out[columns_like]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--additional', required=True)
    parser.add_argument('--out', default='data/processed')
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    main_df = prepare_main(args.main)
    add_df = prepare_additional(args.additional, list(main_df.columns))
    main_df.to_csv(out_dir / 'main_modeling_dataset.csv', index=False)
    add_df.to_csv(out_dir / 'additional_harmonized_dataset.csv', index=False)
    pd.concat([main_df, add_df], ignore_index=True).to_csv(out_dir / 'combined_common_schema_dataset.csv', index=False)
    print(f'Saved datasets to {out_dir}')


if __name__ == '__main__':
    main()



