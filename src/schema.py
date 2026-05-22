MAIN_TARGET = 'is_canceled'

LEAKAGE_COLUMNS = [
    'reservation_status',
    'reservation_status_date',
    'assigned_room_type',
    'booking_changes',
    'days_in_waiting_list',
]

REQUIRED_MAIN_COLUMNS = [
    'hotel','is_canceled','lead_time','arrival_date_year','arrival_date_month',
    'arrival_date_day_of_month','stays_in_weekend_nights','stays_in_week_nights',
    'adults','children','babies','meal','market_segment','distribution_channel',
    'is_repeated_guest','previous_cancellations','previous_bookings_not_canceled',
    'reserved_room_type','deposit_type','customer_type','adr',
    'required_car_parking_spaces','total_of_special_requests'
]



