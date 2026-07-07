# Hotel Booking Cancellation Risk — финальный MVP

ML-сервис для revenue-менеджера отеля. Сервис загружает CSV с бронированиями, проверяет качество схемы, считает вероятность отмены, переводит ее в ожидаемую денежную потерю, формирует список броней к ручной проверке и сохраняет результаты скоринга в СУБД.

Короткий тезис для защиты:

```text
Сервис за 3 минуты превращает CSV бронирований в список заявок, где сконцентрирован денежный риск отмен.
```

Формула продуктовой ценности:

```text
risk_score = P(is_canceled = 1)
booking_value = adr × total_nights
expected_loss = cancellation_probability × booking_value × deposit_loss_factor
business_priority_score = expected_loss
```


## Валюта денежных показателей

Сервис не предполагает, что цены в CSV всегда заданы в рублях. В UI можно выбрать валюту отображения: **RUB ₽, EUR €, USD $, GBP £**. Конвертация не выполняется: `adr`, `booking_value`, `expected_loss` считаются в валюте исходного файла, а выбранная валюта используется только для отображения KPI и таблиц. Подробнее: `docs/CURRENCY_HANDLING.md`.

## Что доработано перед финальной защитой

| Зона | Что изменено |
|---|---|
| Модель | В архиве есть `models/hw8_model.joblib`; `/health` и `/api/v1/model-info` явно показывают `model_status=trained/fallback` |
| Fallback | Fallback больше не скрыт: API возвращает предупреждение, UI показывает красный баннер. Для строгого режима можно задать `HOTEL_RISK_REQUIRE_TRAINED_MODEL=1` |
| Метрики | Добавлен единый `docs/MODEL_CARD.md` с финальными HW8-метриками сервисной модели |
| UI | После анализа сразу открывается «Список к проверке», а не аналитический scatter plot |
| UI-фильтры | Добавлены фильтры по горизонту заезда, отелю, сегменту, каналу, категории риска и минимальной ожидаемой потере |
| Валюта | UI поддерживает RUB ₽, EUR €, USD $, GBP £; конвертация не выполняется |
| CSV quality | Валидация возвращает `quality_status`, `schema_confidence`, `defaulted_ratio`, `quality_message`; UI подсвечивает качество входного файла |
| Объяснения | Локальные причины переименованы в «Бизнес-факторы риска», чтобы не выдавать эвристики за SHAP |
| API | `POST /api/v1/batches/upload?return_predictions=true` теперь одним вызовом сохраняет batch и возвращает predictions для UI |
| Workers | `scoring_jobs` теперь реально используется: job получает статусы `created/queued/scoring/completed/failed`; добавлен `GET /api/v1/jobs/{job_id}` |
| Ограничения | Для sync endpoints добавлены лимиты на размер файла и число строк |
| Технический долг | Убран deprecated FastAPI `on_event`; Pydantic `Config` заменен на `ConfigDict`; CORS ограничен локальными UI origin по умолчанию |
| Упаковка | Добавлены отсутствовавшие файлы сдачи: `README_HW8_SUBMISSION.md`, `docs/HW8_SCREENSHOT_GUIDE.md` |

## Что закрыто по критериям задания

| Требование | Реализация в проекте |
|---|---|
| Доменная модель сервиса | `src/hotel_risk/domain.py`, `src/hotel_risk/db.py` |
| Хранение данных за счет СУБД | SQLite локально, PostgreSQL в Docker Compose; таблицы `scoring_batches`, `bookings`, `predictions`, `scoring_jobs`, `audit_logs` |
| REST API | FastAPI: `src/hotel_risk/api/main.py`; Swagger: `http://127.0.0.1:8000/docs` |
| UI | Streamlit: `ui/streamlit_app.py` |
| Тесты | `tests/`, текущий результат: `60 passed` |
| Docker | `Dockerfile.api`, `Dockerfile.ui`, `Dockerfile.worker`, `docker-compose.yml` |
| Масштабирование воркеров | Redis + RQ worker; команда `docker compose up --scale worker=3` |

## Структура проекта

```text
├── README.md
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── Dockerfile.worker
├── requirements-mvp.txt
├── pyproject.toml
├── data/
│   ├── raw/
│   ├── processed/
│   └── examples/
├── models/
│   └── hw8_model.joblib
├── src/
│   ├── hotel_risk/
│   │   ├── api/main.py
│   │   ├── business.py
│   │   ├── config.py
│   │   ├── db.py
│   │   ├── domain.py
│   │   ├── features.py
│   │   ├── ml.py
│   │   ├── repository.py
│   │   ├── schemas.py
│   │   ├── train.py
│   │   ├── worker.py
│   │   └── worker_jobs.py
│   └── prepare_dataset.py
├── ui/
│   └── streamlit_app.py
├── docs/
│   ├── MODEL_CARD.md
│   ├── DEFENSE_GUIDE.md
│   ├── BUSINESS_ECONOMICS.md
│   └── UI_ROLES.md
└── tests/
```

## Локальный запуск без Docker

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mvp.txt
pip install -e .
```

Проверить, что модельный артефакт есть:

```bash
ls -lh models/hw8_model.joblib
```

Переобучить / переупаковать модель:

```bash
python -m hotel_risk.train \
  --data data/processed/main_modeling_dataset.csv \
  --model models/hw8_model.joblib \
  --metrics reports/tables/hw8_model_metrics.json
```

Запуск API:

```bash
uvicorn hotel_risk.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Запуск UI во втором терминале:

```bash
source .venv/bin/activate
export HOTEL_RISK_API_URL=http://127.0.0.1:8000
streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Адреса:

```text
UI:      http://localhost:8501
API:     http://localhost:8000
Swagger: http://localhost:8000/docs
```

## Демо-авторизация

| Роль | Логин | Пароль | Интерфейс |
|---|---|---|---|
| Менеджер | `user` | `user` | Загрузка данных, запуск анализа, KPI, список к проверке, карточка брони, история |
| Администратор | `admin` | `admin` | Статус API, модель, СУБД, схема API, пользователи, логи |

На странице входа есть регистрация нового пользователя. Зарегистрированный аккаунт автоматически получает роль менеджера. Для учебного MVP пароли зарегистрированных пользователей сохраняются как PBKDF2-SHA256 hash с солью в `storage/users.json`. Администраторский доступ через регистрацию не выдается.

## Запуск через Docker Compose

```bash
docker compose up --build
```

Масштабирование воркеров:

```bash
docker compose up --build --scale worker=3
```

В Docker Compose используется PostgreSQL:

```text
postgresql+psycopg2://hotel_risk:hotel_risk@postgres:5432/hotel_risk
```

Локально по умолчанию используется SQLite:

```text
storage/hotel_risk.db
```

## REST API

Основные ручки:

```text
GET  /health
GET  /api/v1/domain-model
GET  /api/v1/model-info
GET  /api/v1/schema
GET  /api/v1/sample-csv?format=canonical
GET  /api/v1/sample-csv?format=flexible
POST /api/v1/validate
POST /api/v1/validate-csv
POST /api/v1/score
POST /api/v1/score-csv
POST /api/v1/batches/upload?return_predictions=true
GET  /api/v1/batches
GET  /api/v1/batches/{batch_id}
GET  /api/v1/batches/{batch_id}/predictions
GET  /api/v1/batches/{batch_id}/insights
GET  /api/v1/batches/{batch_id}/export
GET  /api/v1/analytics/overview
POST /api/v1/jobs/score-file
GET  /api/v1/jobs/{job_id}
```

## CSV формат

Сервис поддерживает два режима:

1. Канонический CSV с колонками исходного Hotel Booking Demand Dataset.
2. Flexible CSV: сервис пытается распознать похожие названия колонок и заполнить недостающие поля безопасными дефолтами.

Примеры:

```text
data/examples/canonical_upload_example.csv
data/examples/flexible_upload_example.csv
```

Если слишком много колонок заполнено дефолтами, API вернет `quality_status=red`, а UI покажет предупреждение. Такой результат годится только для технической демонстрации, не для бизнес-решения.

## Финальные метрики сервисной модели

Финальная модель в MVP: `LightGBM FE TE conservative + Platt calibration`.

```text
ROC-AUC       = 0.8595
PR-AUC        = 0.6791
Brier         = 0.1409
Precision@20  = 0.6892
Recall@20     = 0.4874
Lift@20       = 2.4375
```

Главная метрика для питча — `Lift@20`: верхние 20% броней по риску содержат отмены примерно в 2.44 раза чаще, чем случайный список такого же размера. Подробности: `docs/MODEL_CARD.md`.

## Тесты

```bash
pytest -q
```

Текущий результат после доработок:

```text
60 passed
```

## Переменные окружения

| Переменная | Значение по умолчанию | Назначение |
|---|---|---|
| `HOTEL_RISK_API_URL` | `http://127.0.0.1:8000` | URL API для Streamlit |
| `HOTEL_RISK_DATABASE_URL` | `sqlite:///./storage/hotel_risk.db` | Подключение к СУБД |
| `HOTEL_RISK_MODEL_PATH` | `models/hw8_model.joblib` | Путь к модели |
| `HOTEL_RISK_REQUIRE_TRAINED_MODEL` | `0` | Если `1`, API не стартует без trained artifact |
| `HOTEL_RISK_ALLOW_FALLBACK_MODEL` | `1` | Разрешает demo rule-based fallback |
| `HOTEL_RISK_SYNC_MAX_ROWS` | `1000000` | Лимит строк для sync scoring |
| `HOTEL_RISK_MAX_UPLOAD_BYTES` | `31457280` | Лимит размера CSV |
| `HOTEL_RISK_CORS_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Разрешенные UI origins |
