# Домашнее задание №7. Улучшаем модель

## Проект

**Hotel Booking Cancellation Risk** — ML-сервис для revenue-менеджера отеля. Сервис заранее оценивает вероятность отмены бронирования, ранжирует бронирования по уровню риска и переводит ML-score в бизнес-приоритет.

Продуктовая цепочка остается такой же, как в ДЗ №5–6:

```text
данные бронирования
→ прогноз вероятности отмены
→ оценка ожидаемой финансовой потери
→ бизнес-приоритизация
→ рекомендованное действие менеджера
```

В ДЗ №7 задача — улучшить baseline-модель и подготовить решение к MVP. Улучшение сделано не через leakage, а через расширение признакового пространства, train-only encoding и аккуратную постобработку предсказаний.

---

## 1. Что было в ДЗ №5–6

В предыдущем ноутбуке лучшей моделью стала **LightGBM random search**.

| Модель | quality_index | ROC-AUC | PR-AUC | Brier | Precision@20 | Recall@20 | Lift@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM random search | 0.7607 | 0.8530 | 0.7942 | 0.1607 | 0.8172 | 0.4147 | 2.0730 |

Главная продуктовая логика ДЗ №5–6 была корректной: использовать не только вероятность отмены, но и финансовый приоритет:

```text
risk_score = P(is_canceled = 1)
expected_loss = risk_score × booking_value × deposit_loss_factor
business_priority_score = expected_loss
```

В ДЗ №7 эта логика сохранена и усилена.

---

## 2. Что улучшено в ДЗ №7

### 2.1. Пайплайн предобработки и feature engineering

В ноутбуке `07_model_improvement.ipynb` реализован полный pipeline:

1. Загрузка `data/raw/hotel_bookings.csv`.
2. Очистка пропусков и невалидных значений.
3. Удаление бронирований без гостей.
4. Удаление отрицательного `adr`.
5. Создание `arrival_date`.
6. Time-based split на train / validation / test.
7. Leakage audit.
8. Генерация новых признаков.
9. Train-only encoding.
10. Preprocessing через `ColumnTransformer`.

Базовые признаки из ДЗ №5–6 сохранены:

| Признак | Логика |
|---|---|
| `total_nights` | общая длительность проживания |
| `total_guests` | количество гостей |
| `booking_value` | `adr × total_nights` |
| `has_children` | есть дети или младенцы |
| `has_special_requests` | есть специальные запросы |
| `has_previous_cancellations` | были предыдущие отмены |
| `lead_time_bin` | категория глубины бронирования |
| `nights_bin` | категория длительности проживания |
| `adr_bin` | категория стоимости ночи |

В ДЗ №7 добавлены новые признаки:

| Группа | Признаки | Зачем |
|---|---|---|
| Calendar features | `arrival_quarter`, `arrival_dayofweek`, `arrival_is_weekend` | сезонность и день недели |
| Cyclical features | `month_sin`, `month_cos`, `week_sin`, `week_cos` | цикличность месяца и недели без ложного порядка |
| Ratio features | `weekend_share`, `adr_per_guest` | структура пребывания и цена на гостя |
| Interaction features | `lead_time_x_value`, `lead_time_x_requests` | взаимодействие глубины брони со стоимостью и запросами |
| Log features | `lead_time_log`, `adr_log`, `value_per_night_log` | снижение влияния выбросов |
| Behavioral flags | `is_single_adult`, `is_family`, `has_parking_request`, `no_special_no_parking` | интерпретируемые паттерны бронирований |

Синтетическая генерация новых строк не применялась. Для этой задачи она менее надежна, потому что бронирования имеют временную структуру, сезонность и финансовую интерпретацию. Неправильная генерация могла бы улучшить offline-метрики, но ухудшить калибровку и production-поведение. Поэтому вместо synthetic data выбран feature engineering и безопасное кодирование категорий.

---

## 3. Leakage audit

В ДЗ №7 сохранен запрет на признаки, которые могут быть известны только после отмены, после заезда или после операционных изменений бронирования.

| Признак | Решение | Причина |
|---|---|---|
| `reservation_status` | exclude | содержит финальный статус бронирования |
| `reservation_status_date` | exclude | содержит дату финального статуса / отмены |
| `assigned_room_type` | exclude | может быть известен после операционного назначения номера |
| `booking_changes` | exclude | может изменяться после создания бронирования |

Отличие от ДЗ №5–6 касается `agent` и `company`.

В ДЗ №5–6 они были исключены полностью как high-cardinality operational ID. В ДЗ №7 raw ID по-прежнему не используются как категориальные признаки. Вместо этого из них построены только train-only признаки:

```text
agent_freq_log
agent_target_mean_smooth
company_freq_log
company_target_mean_smooth
```

Target encoding считается только на train. Validation и test не участвуют в расчете статистик. Неизвестные категории получают глобальный train prior. Поэтому это не является прямым leakage.

---

## 4. Улучшенная архитектура модели

Финальная модель:

```text
LightGBMClassifier
+ expanded feature set
+ train-only frequency / target encoding
+ class_weight='balanced'
+ регуляризация reg_lambda / reg_alpha
+ validation-based model selection
```

Были проверены две конфигурации LightGBM. Лучшая выбрана по validation ROC-AUC.

Финальные параметры лучшей модели:

```python
{
    "num_leaves": 64,
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 260,
    "min_child_samples": 70,
    "subsample": 0.90,
    "colsample_bytree": 0.80,
    "reg_lambda": 10.0,
    "reg_alpha": 0.5,
}
```

Важно: test split не использовался для выбора параметров или threshold.

---

## 5. Постобработка предсказаний

Постобработка состоит из двух уровней.

### 5.1. ML-постобработка

На validation выбирается threshold по F2-score. F2 выбран потому, что для revenue-менеджера пропустить потенциальную отмену обычно хуже, чем отправить в обработку лишнюю бронь.

Также проверена isotonic calibration. Она немного улучшает Brier, но ухудшает ranking-качество (`PR-AUC` и `quality_index`). Поэтому для ранжирования используется raw LightGBM score, а calibrated score можно оставить только для probability display.

### 5.2. Бизнес-постобработка

Модель возвращает `risk_score`, но MVP должен сортировать бронирования по бизнес-приоритету:

```text
booking_value = adr × total_nights
expected_loss = risk_score × booking_value × deposit_loss_factor
business_priority_score = expected_loss
```

Коэффициенты депозитов:

| deposit_type | deposit_loss_factor | Логика |
|---|---:|---|
| `No Deposit` | 1.00 | отель слабо защищен, отмена может привести к полной потере потенциальной выручки |
| `Refundable` | 0.80 | риск частично сохраняется, особенно если отмена поздняя |
| `Non Refund` | 0.05 | вероятность отмены может быть высокой, но финансовая потеря для отеля существенно ниже |

Также добавлены категории риска:

| Категория | Условие |
|---|---:|
| `Critical` | `risk_score >= 0.80` |
| `High` | `risk_score >= 0.60` |
| `Medium` | `risk_score >= 0.30` |
| `Low` | `< 0.30` |

Для каждой брони формируется `recommended_action`.

---

## 6. Финальные метрики

Финальная ranking-модель — **HW7 LightGBM FE + train-only encoding**.

| Модель | quality_index | ROC-AUC | PR-AUC | Brier | Precision@20 | Recall@20 | Lift@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HW7 LightGBM FE + train-only encoding | 0.7816 | 0.8771 | 0.8197 | 0.1427 | 0.8367 | 0.4246 | 2.1226 |
| HW5–6 LightGBM random search | 0.7607 | 0.8530 | 0.7942 | 0.1607 | 0.8172 | 0.4147 | 2.0730 |

Изменение относительно ДЗ №5–6:

| Метрика | Было | Стало | Изменение |
|---|---:|---:|---:|
| quality_index | 0.7607 | 0.7816 | +0.0209 |
| ROC-AUC | 0.8530 | 0.8771 | +0.0241 |
| PR-AUC | 0.7942 | 0.8197 | +0.0255 |
| Brier | 0.1607 | 0.1427 | -0.0180 |
| Precision@20 | 0.8172 | 0.8367 | +0.0195 |
| Recall@20 | 0.4147 | 0.4246 | +0.0099 |
| Lift@20 | 2.0730 | 2.1226 | +0.0496 |

Brier уменьшился, что является улучшением, потому что для Brier меньше — лучше.

---

## 7. Интерпретация результата

`Lift@20 = 2.1226` означает, что верхние 20% бронирований, выбранные моделью, содержат отмены примерно в 2.12 раза чаще, чем случайная выборка такого же размера.

`Precision@20 = 0.8367` означает, что среди верхних 20% бронирований по риску около 83.6% действительно относятся к отменам.

Это соответствует продуктовой логике: revenue-менеджер не должен смотреть всю базу бронирований. Он должен получить ограниченный top-list, где концентрация потенциальных отмен значительно выше среднего.

Главное улучшение по сравнению с ДЗ №5–6:

```text
модель стала лучше ранжировать риск отмены
и при этом лучше калибровать вероятность отмены.
```

---

## 8. Анализ ошибок

В ноутбуке добавлен отдельный блок анализа ошибок:

1. Confusion matrix по threshold, выбранному на validation.
2. Segment quality по:
   - `hotel`,
   - `deposit_type`,
   - `market_segment`,
   - `customer_type`.
3. Таблица false positives.
4. Таблица false negatives.
5. Ошибки в разрезе `hotel × deposit_type × error_type`.

Это важно для MVP, потому что средняя метрика может скрывать слабые места. Например, модель может хорошо работать на `City Hotel`, но хуже на `Resort Hotel`, или давать высокий risk score для `Non Refund`, который финансово не является top-priority.

---

## 9. Что сохраняет ноутбук

После запуска ноутбук сохраняет артефакты в `reports/tables` и `reports/figures/hw7_model_improvement`.

Ключевые таблицы:

| Файл | Назначение |
|---|---|
| `hw7_model_improvement_leaderboard.csv` | сравнение HW5–6, HW7 raw, HW7 calibrated |
| `hw7_metric_diff_vs_hw56.csv` | изменение метрик относительно предыдущего baseline |
| `hw7_top_business_priority_bookings.csv` | top бронирований для revenue-менеджера |
| `hw7_test_predictions_for_mvp.csv` | test-score выгрузка для MVP |
| `hw7_segment_quality.csv` | качество по сегментам |
| `hw7_error_by_segment.csv` | ошибки по сегментам |
| `hw7_false_positives_examples.csv` | примеры false positive |
| `hw7_false_negatives_examples.csv` | примеры false negative |
| `hw7_final_summary.json` | краткий machine-readable итог |

---

## 10. Итоговый вывод

Финальная модель ДЗ №7 — **HW7 LightGBM FE + train-only encoding**.

Она улучшила все ключевые метрики относительно ДЗ №5–6:

```text
quality_index: 0.7607 → 0.7816
ROC-AUC:       0.8530 → 0.8771
PR-AUC:        0.7942 → 0.8197
Brier:         0.1607 → 0.1427
Precision@20:  0.8172 → 0.8367
Recall@20:     0.4147 → 0.4246
Lift@20:       2.0730 → 2.1226
```

