# Локальный запуск MVP

Для ДЗ №8 публичный адрес не нужен. Достаточно локального запуска и ссылки на GitHub-репозиторий.

## API

```bash
uvicorn hotel_risk.api.main:app --host 127.0.0.1 --port 8000 --reload
```

## UI

```bash
export HOTEL_RISK_API_URL=http://127.0.0.1:8000
streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Открывать:

```text
http://localhost:8501
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Streamlit может показывать `Network URL` или `External URL`. Для этого задания их использовать не нужно.
