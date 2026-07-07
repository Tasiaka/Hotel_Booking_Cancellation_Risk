# Model Card — Hotel Booking Cancellation Risk MVP

## Назначение модели

Модель оценивает вероятность отмены гостиничного бронирования и помогает revenue-менеджеру получить приоритетный список заявок для ручной проверки. В продукте вероятность не используется как автоматическое решение. Она переводится в финансовый приоритет:

```text
risk_score = P(is_canceled = 1)
booking_value = adr × total_nights
expected_loss = cancellation_probability × booking_value × deposit_loss_factor
business_priority_score = expected_loss
```

## Финальный артефакт

```text
models/hw8_model.joblib
```

Модель, сохраненная в сервисном артефакте:

```text
LightGBM FE TE conservative + Platt calibration
```

Сервисные endpoints `/health` и `/api/v1/model-info` показывают статус артефакта:

```text
model_status = trained | fallback
```

Если trained artifact отсутствует, API может включить rule-based fallback только для demo/local режима. Для защиты и пилота используйте `HOTEL_RISK_REQUIRE_TRAINED_MODEL=1`.

## Данные

Основной датасет: Hotel Booking Demand Dataset.

Объект наблюдения: одно бронирование.

Целевая переменная:

```text
is_canceled
0 — бронирование состоялось
1 — бронирование отменено / no-show
```

Момент прогноза: после создания бронирования, но до отмены, до заезда и до финального статуса. Поэтому модель исключает post-factum признаки.


## Split-стандарт

Финальный сервисный контур использует единый стандарт из ДЗ №4: разбиение по расчетной дате создания бронирования `booking_creation_date = arrival_date - lead_time`, а не по дате заезда и не random split. Это ближе к production-сценарию: модель обучается на бронированиях, созданных раньше, и применяется к бронированиям, созданным позже.

```text
train: booking_creation_date <= 2016-10-31
valid: 2016-11-01 <= booking_creation_date <= 2017-03-31
test:  booking_creation_date > 2017-03-31
```

В `src/prepare_dataset.py` этот split формируется при подготовке `data/processed/main_modeling_dataset.csv`. В `src/hotel_risk/ml.py` тот же стандарт применяется автоматически, если обучающий CSV не содержит готовой колонки `split`. В метриках модели это зафиксировано полями `split_strategy` и `split_standard`.

## Leakage audit

Исключены признаки:

| Признак | Причина |
|---|---|
| `reservation_status` | напрямую кодирует итоговый статус бронирования |
| `reservation_status_date` | содержит дату финального статуса / отмены |
| `assigned_room_type` | может быть известен после операционного назначения номера |
| `booking_changes` | может изменяться после создания бронирования |
| `days_in_waiting_list` | не гарантированно известен в момент прогноза |

`agent` и `company` не используются как raw high-cardinality категории. Из них построены только train-only frequency/target-encoding признаки. Validation/test/inference не участвуют в расчете этих статистик.

## Feature engineering

Основные группы признаков:

| Группа | Примеры |
|---|---|
| Базовые booking features | `lead_time`, `hotel`, `market_segment`, `distribution_channel`, `deposit_type`, `customer_type`, `adr` |
| Стоимость | `total_nights`, `booking_value`, `adr_per_guest` |
| Календарь | `arrival_quarter`, `arrival_dayofweek`, `month_sin`, `month_cos`, `week_sin`, `week_cos` |
| Поведение | `has_previous_cancellations`, `has_special_requests`, `has_parking_request`, `is_repeated_guest` |
| Взаимодействия | `lead_time_x_value`, `lead_time_x_requests` |
| Train-only encoding | `agent_freq_log`, `agent_target_mean_smooth`, `country_target_mean_smooth`, `market_segment_target_mean_smooth` |

## Финальные offline-метрики сервисной модели

Файл с метриками:

```text
reports/tables/hw8_model_metrics.json
```

| Метрика | Значение | Интерпретация |
|---|---:|---|
| ROC-AUC | 0.8595 | модель хорошо разделяет отмены и состоявшиеся брони |
| PR-AUC | 0.6791 | качество на положительном классе отмен |
| Brier | 0.1409 | калибровка вероятности после Platt calibration |
| Precision@20 | 0.6892 | среди верхних 20% по риску около 68.9% — фактические отмены |
| Recall@20 | 0.4874 | верхние 20% покрывают около 48.7% всех отмен на test |
| Lift@20 | 2.4375 | top-20% список лучше случайного примерно в 2.44 раза |

Главная метрика для защиты — `Lift@20`, потому что revenue-менеджер работает с ограниченным списком броней, а не со всей базой.

## Почему HW8-метрики отличаются от HW7

В исследовательском ДЗ №7 метрики были выше на другом экспериментальном контуре. В MVP показывается сервисная HW8-модель, потому что именно она сохранена в `models/hw8_model.joblib` и используется FastAPI/Streamlit. В защите нужно показывать одну финальную таблицу HW8, а HW7 упоминать только как исследовательскую итерацию.

## Ограничения

1. Данные исторические, 2015–2017 годы. Реальный отель может иметь другой cancellation pattern.
2. `Canceled` и `No-Show` объединены в один положительный класс, хотя управленческие действия могут отличаться.
3. Нет timestamp отмены, поэтому модель не различает ранние и поздние отмены.
4. Нет данных об интервенциях менеджера, поэтому protected revenue — сценарная оценка, не доказанный causal uplift.
5. Flexible CSV с большим числом дефолтов снижает надежность. API и UI теперь явно показывают `quality_status`.

## Что проверять в пилоте

1. Переносимость модели на данные конкретного отеля.
2. Калибровку вероятностей после 2–4 недель live-scoring.
3. Раздельные метрики по hotel type, market segment, channel, deposit type.
4. Реальный эффект действий менеджера: подтверждение брони, депозит, тариф, ручная проверка.
5. Drift cancellation rate и доли каналов продаж.
