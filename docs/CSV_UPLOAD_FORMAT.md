# Формат CSV для загрузки

В MVP можно загрузить CSV с бронированиями двумя способами.

## 1. Полный формат

Пример находится здесь:

```text
data/examples/canonical_upload_example.csv
```

Основные колонки:

```text
booking_id, hotel, lead_time, arrival_date_year, arrival_date_month,
arrival_date_day_of_month, arrival_date_week_number,
stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
meal, country, market_segment, distribution_channel, is_repeated_guest,
previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
deposit_type, customer_type, adr, required_car_parking_spaces,
total_of_special_requests
```

## 2. Гибкий формат

Пример находится здесь:

```text
data/examples/flexible_upload_example.csv
```

Такой файл может использовать альтернативные названия колонок, например:

```text
hotel_type, LeadTime, arrival_date, no_of_weekend_nights,
no_of_week_nights, no_of_adults, no_of_children,
market_segment_type, booking_channel, room_type_reserved,
avg_price_per_room
```

Сервис приводит такие поля к внутренней схеме автоматически. Лишние поля не участвуют в прогнозе.

## Денежные показатели

В интерфейсе используется термин **ожидаемая потеря**. Это не вся стоимость брони, а ожидаемая сумма потери:

```text
ожидаемая потеря =
калиброванная вероятность отмены × стоимость брони × финансовая поправка только для невозвратного тарифа
```

Для `Non Refund` коэффициент финансовой потери равен нулю, поэтому такие брони могут иметь высокую вероятность отмены, но не попадать в рабочий список менеджера, если денежный ущерб мал.
