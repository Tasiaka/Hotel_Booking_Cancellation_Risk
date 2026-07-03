# Hotel Booking Cancellation Risk — MVP для ДЗ №8

MVP-сервис для revenue-менеджера отеля: сервис загружает CSV с бронированиями, проверяет схему, рассчитывает риск отмены, показывает выручку под риском и сохраняет результаты скоринга в СУБД.

Проект подготовлен под домашнее задание №8 «Упаковка MVP».

## Что закрыто по критериям задания

| Требование | Реализация в проекте |
|---|---|
| Доменная модель сервиса | `src/hotel_risk/domain.py`, `src/hotel_risk/db.py`, отчет `homework_08/packaging_mvp_report.md` |
| Хранение данных за счет СУБД | SQLite локально, PostgreSQL в Docker Compose; таблицы batch, bookings, predictions, jobs, audit logs |
| REST интерфейс | FastAPI: `src/hotel_risk/api/main.py`, Swagger: `http://127.0.0.1:8000/docs` |
| Пользовательский интерфейс | Streamlit: `ui/streamlit_app.py` |
| Тесты критических частей | `tests/`, ожидаемый результат: `59 passed` |
| Docker контейнер | `Dockerfile.api`, `Dockerfile.ui`, `Dockerfile.worker`, `docker-compose.yml` |
| Масштабирование воркеров | Redis + worker; команда `docker compose up --scale worker=3` |

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
│   │   ├── db.py
│   │   ├── domain.py
│   │   ├── features.py
│   │   ├── ml.py
│   │   ├── repository.py
│   │   ├── schemas.py
│   │   ├── train.py
│   │   └── worker.py
│   └── prepare_dataset.py
├── ui/
│   └── streamlit_app.py
├── tests/
└── docs/
```

## Локальный запуск без Docker

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mvp.txt
pip install -e .
```

Обучение / переупаковка модели в бинарный артефакт:

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

Открыть UI:

```text
http://localhost:8501
```

Открыть Swagger:

```text
http://127.0.0.1:8000/docs
```


## Логин и регистрация UI

Streamlit-интерфейс разделен на роли:

```text
user / user       рабочая панель менеджера
admin / admin     техническая админская часть
```

На странице входа есть регистрация нового пользователя. Зарегистрированный аккаунт автоматически получает роль менеджера. Локальное хранилище пользователей:

```text
storage/users.json
```

Для учебного MVP пароли зарегистрированных пользователей сохраняются не в открытом виде, а как PBKDF2-SHA256 hash с солью. Администраторский доступ не выдается через саморегистрацию; роль admin остается только у `admin/admin`.

В роли менеджера интерфейс специально упрощен: одна основная кнопка «Запустить анализ», итоговые KPI, список броней к проверке, карточка брони и история. Технические детали, REST endpoints, схема API и логи вынесены в роль администратора.

## Запуск через Docker Compose

```bash
docker compose up --build
```

Адреса:

```text
UI:      http://localhost:8501
API:     http://localhost:8000
Swagger: http://localhost:8000/docs
```

Масштабирование воркеров с моделью:

```bash
docker compose up --build --scale worker=3
```

## СУБД

Локально используется SQLite:

```text
storage/hotel_risk.db
```

В Docker Compose используется PostgreSQL:

```text
postgresql+psycopg2://hotel_risk:hotel_risk@postgres:5432/hotel_risk
```

Сервис сохраняет:

```text
hotel_properties      объект размещения
scoring_batches       пакет скоринга
bookings              нормализованные бронирования
predictions           прогнозы модели
scoring_jobs          задания воркеров
audit_logs            журнал событий
```

Главная ручка для сохранения результатов в СУБД:

```text
POST /api/v1/batches/upload
```

## REST API

Основные ручки:

```text
GET  /health
GET  /api/v1/domain-model
GET  /api/v1/model-info
GET  /api/v1/schema
POST /api/v1/validate
POST /api/v1/validate-csv
POST /api/v1/score
POST /api/v1/score-csv
POST /api/v1/batches/upload
GET  /api/v1/batches
GET  /api/v1/batches/{batch_id}
GET  /api/v1/batches/{batch_id}/predictions
GET  /api/v1/batches/{batch_id}/export
POST /api/v1/jobs/score-file
```

## CSV формат

Сервис поддерживает два режима:

1. Канонический CSV с колонками исходного датасета.
2. Flexible CSV: сервис пытается распознать похожие названия колонок и заполнить недостающие поля безопасными значениями по умолчанию.

Примеры находятся здесь:

```text
data/examples/canonical_upload_example.csv
data/examples/flexible_upload_example.csv
```

## Метрики MVP-модели

Финальная сервисная модель: LightGBM FE + train-only target/frequency encoding + Platt calibration.

```text
ROC-AUC      = 0.8595
PR-AUC       = 0.6791
Brier        = 0.1409
Precision@20 = 0.6892
Recall@20    = 0.4874
Lift@20      = 2.4375
```

Top-20% бронирований по риску содержит отмены примерно в 2.44 раза чаще, чем случайный список такого же размера.

## Тесты

```bash
pytest
```

```text
59 passed
```

## Что приложить в GitHub

В GitHub нужно загрузить весь проект и приложить ссылку на репозиторий. Для проверки удобно открыть:

```text
README.md
README_HW8_SUBMISSION.md
homework_08/packaging_mvp_report.md
docs/HW8_SCREENSHOT_GUIDE.md
```


## Демо-авторизация и роли

В Streamlit UI добавлен login-screen с разделением ролей:

| Роль | Логин | Пароль | Интерфейс |
|---|---|---|---|
| Пользователь | `user` | `user` | Упрощенная рабочая панель менеджера: данные, анализ, KPI, список к проверке, карточка брони, история |
| Администратор | `admin` | `admin` | Админская часть: статус API, модель, СУБД/batch, схема API, логи, плюс обычный пользовательский сценарий |

Авторизация демонстрационная и нужна для учебного MVP. Для реального продукта ее нужно заменить на серверную авторизацию.

## Обновленный UI/UX

Интерфейс упрощен: убрана перегруженная боковая панель, оставлена одна основная кнопка **«Запустить анализ»**, технические детали перенесены в админскую роль, а подробные пояснения открываются в отдельных раскрывающихся блоках.

Документ по ролям и UX:

```text
docs/UI_ROLES.md
```

## UI/UX и роли

В Streamlit добавлена демонстрационная авторизация с двумя ролями:

```text
user / user   — рабочий интерфейс менеджера
admin / admin — административная часть
```

Страница входа сделана адаптивной: верхний hero-блок масштабируется относительно окна, а форма входа находится в компактной центральной карточке. Менеджерская панель упрощена: вместо большого числа кнопок оставлена одна основная кнопка **«Запустить анализ»**, а подробности спрятаны в раскрывающиеся блоки. Технические детали доступны только роли `admin`.


## Бизнес-финансы в интерфейсе

Менеджерский интерфейс больше не требует вручную выбирать долю бронирований к проверке. Сервис автоматически подбирает список с максимальным ожидаемым эффектом: ожидаемо условно сохранено − стоимость проверки. Подробная логика описана в `docs/BUSINESS_ECONOMICS.md`.
