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

Главный файл: [`homework_04/data_understanding_and_dataset.md`](homework_04/data_understanding_and_dataset.md).

В нем описаны:

- источник и состав основного и дополнительного датасетов;
- базовый EDA с выводами, важными для моделирования;
- оценка качества разметки и план ее улучшения;
- алгоритм формирования модельного датасета;
- стратегия временной валидации;
- подготовленные CSV для дальнейшего моделирования.

## Датасеты

- Основной: Hotel Booking Demand Dataset — https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
- Дополнительный: Hotel Reservations Classification Dataset — https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset

Подготовленные данные находятся в `data/processed/`:

```text
data/processed/
├── main_modeling_dataset.csv
├── additional_harmonized_dataset.csv
└── combined_common_schema_dataset.csv
```

## Быстрый запуск подготовки данных

```bash
pip install -r requirements.txt
python src/prepare_dataset.py   --main data/raw/hotel_bookings.csv   --additional "data/raw/Hotel Reservations.csv"   --out data/processed
```

## Структура

```text
hotel-booking-cancellation-risk-hw4/
├── README.md
├── homework_01/
├── homework_02/
├── homework_03/
├── homework_04/
│   └── data_understanding_and_dataset.md
├── notebooks/
│   └── 04_data_understanding_dataset.ipynb
├── src/
│   ├── prepare_dataset.py
│   ├── schema.py
│   └── sanity_check_model.py
├── data/
│   ├── README.md
│   └── processed/
├── reports/
│   ├── figures/
│   └── tables/
└── requirements.txt
```

## Ключевой принцип

Модельный датасет формируется под момент прогноза: после создания бронирования, но до отмены/заезда/финального статуса. Поэтому из признаков исключаются поля с post-factum информацией: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes`, `days_in_waiting_list`.
