# Хранение данных в СУБД

## 1. Назначение

MVP сохраняет результаты скоринга, чтобы менеджер мог вернуться к истории запусков и выгрузить уже рассчитанные прогнозы.

## 2. Локальный режим

При локальном запуске используется SQLite:

```text
storage/hotel_risk.db
```

SQLite удобен для учебной проверки: не нужно отдельно поднимать сервер БД.

## 3. Docker Compose режим

В Docker Compose используется PostgreSQL:

```text
postgresql+psycopg2://hotel_risk:hotel_risk@postgres:5432/hotel_risk
```

Данные хранятся в volume:

```text
pgdata:/var/lib/postgresql/data
```

## 4. Таблицы

| Таблица | Назначение |
|---|---|
| `hotel_properties` | объект размещения |
| `scoring_batches` | запуск скоринга |
| `bookings` | нормализованные бронирования |
| `predictions` | прогнозы модели |
| `scoring_jobs` | задания worker-слоя |
| `audit_logs` | события сервиса |

## 5. Главная ручка сохранения

```text
POST /api/v1/batches/upload
```

После скоринга UI вызывает эту ручку и сохраняет batch в БД. История отображается во вкладке «История скорингов».
