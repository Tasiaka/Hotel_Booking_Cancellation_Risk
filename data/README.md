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
