# Данные

## Источники

1. `hotel_bookings.csv` — основной датасет Hotel Booking Demand Dataset. Источник: Kaggle `jessemostipak/hotel-booking-demand`, исходная публикация Data in Brief `Hotel booking demand datasets`.
2. `Hotel Reservations.csv` — дополнительный датасет Hotel Reservations Classification Dataset. Источник: Kaggle `ahsan81/hotel-reservations-classification-dataset`.

В папке `data/processed/` лежат подготовленные датасеты для ДЗ №4.

## Подготовленные файлы

- `main_modeling_dataset.csv` — основной очищенный датасет для обучения и валидации модели отмен бронирований. Включает engineered-признаки, целевую переменную `is_canceled`, колонку `split` и исключает признаки с утечкой.
- `additional_harmonized_dataset.csv` — дополнительный датасет, приведенный к общей схеме. Предназначен для внешней проверки устойчивости и анализа переносимости, не для слепого объединения с основным датасетом.
- `combined_common_schema_dataset.csv` — объединенная таблица с признаком `source_dataset`; используется для сравнительного анализа и экспериментов с domain-aware validation.

## Исключенные признаки из основного датасета

- `reservation_status`, `reservation_status_date` — прямой leakage итогового статуса.
- `assigned_room_type` — может быть известен после операционного распределения номера.
- `booking_changes` — может изменяться после создания бронирования.
- `days_in_waiting_list` — неоднозначен по моменту доступности.
- `company` — 94.3% пропусков; заменен на бинарный признак `has_company`.

## Базовая стратегия разбиения

Разбиение делается по `booking_creation_date = arrival_date - lead_time`:

- train: до 2016-10-31 включительно;
- validation: 2016-11-01 — 2017-03-31;
- test: 2017-04-01 и позже.

Такой split ближе к реальному сценарию, чем random split: модель обучается на прошлом и проверяется на будущих бронированиях


## CSV для MVP-интерфейса

Лучший формат загрузки — одна строка = одно бронирование. Канонический пример лежит в:

```text
data/examples/canonical_upload_example.csv
```

Минимально рекомендуемые колонки:

```text
booking_id, hotel, lead_time, arrival_date_year, arrival_date_month,
arrival_date_day_of_month, arrival_date_week_number,
stays_in_weekend_nights, stays_in_week_nights, adults, children, babies,
meal, country, market_segment, distribution_channel, is_repeated_guest,
previous_cancellations, previous_bookings_not_canceled, reserved_room_type,
deposit_type, customer_type, adr, required_car_parking_spaces,
total_of_special_requests
```

Начиная с версии HW8-flexcsv сервис также принимает неидеальные CSV. Он умеет:

- брать только нужные для модели колонки;
- игнорировать лишние колонки;
- удалять leakage-колонки `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`;
- распознавать частые альтернативные названия колонок, например `avg_price_per_room → adr`, `no_of_adults → adults`, `arrival_date → arrival_date_year/month/day/week`;
- заполнять отсутствующие признаки безопасными дефолтами.

Пример неканонического файла:

```text
data/examples/flexible_upload_example.csv
```

Важно: полностью произвольный датасет без смысла бронирований невозможно честно скорить. Если в файле нет хотя бы признаков вроде `lead_time`, даты заезда, числа ночей, гостей и цены, сервис вернет best-effort оценку на дефолтах. Такой результат подходит для демонстрации устойчивости MVP, но не для реального бизнес-решения.
