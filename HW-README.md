# Hotel Booking Cancellation Risk

Проект MFDP: ML-сервис для прогнозирования отмен гостиничных бронирований.

## Цель проекта

Разработать прототип сервиса для revenue-менеджера отеля, который оценивает вероятность отмены бронирования и помогает заранее принять управленческие решения: отправить напоминание, запросить подтверждение, учесть риск при планировании загрузки или применить ручную проверку.

## Датасет

Используется публичный датасет Hotel Booking Demand Dataset.

Источник:
- https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/

## Структура репозитория

```text
hotel-booking-cancellation-risk/
├── README.md
└── homework_01/
    └── business_understanding.md
```

# Домашние задания

- `homework_01/business_understanding.md` — бизнес-анализ проекта, выбор датасета, постановка ML-задачи, риски, метрики и план проекта.


## ДЗ №4


- `homework_04/data_understanding_and_dataset.md` — отчет с описанием источников, состава данных, EDA, качества разметки, алгоритма формирования выборки и стратегии валидации.
- `notebooks/04_data_understanding_dataset.ipynb` — воспроизводимый ноутбук с кодом EDA, таблицами, визуализациями, leakage audit и формированием итоговых датасетов.

## Датасеты

- Основной: Hotel Booking Demand Dataset — https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Дополнительный: Hotel Reservations Classification Dataset — https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

Подготовленные данные находятся в `data/processed/`:

```text
data/processed/
├── main_modeling_dataset.csv
├── additional_harmonized_dataset.csv - основной датасет для modeling
└── combined_common_schema_dataset.csv - объединенный датасет в общей схеме
```

Дополнительные артефакты:

- `reports/figures/hw4_eda/` — графики EDA для проверки фактуры анализа.
- `reports/tables/hw4_*.csv` — таблицы EDA и проверок качества данных.


## Ключевой принцип

Модельный датасет формируется под момент прогноза: после создания бронирования, но до отмены/заезда/финального статуса. Поэтому из признаков исключаются поля с post-factum информацией: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`.



---
