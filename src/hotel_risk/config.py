from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
from pydantic import BaseModel, ConfigDict, Field


def _bool_env(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_env(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


class Settings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str = "Hotel Booking Cancellation Risk"
    environment: str = Field(default_factory=lambda: os.getenv("HOTEL_RISK_ENV", "local"))
    database_url: str = Field(default_factory=lambda: os.getenv("HOTEL_RISK_DATABASE_URL", "sqlite:///./storage/hotel_risk.db"))
    redis_url: str = Field(default_factory=lambda: os.getenv("HOTEL_RISK_REDIS_URL", "redis://redis:6379/0"))
    model_path: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_MODEL_PATH", "models/hw8_model.joblib")))
    metrics_path: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_METRICS_PATH", "reports/tables/hw8_model_metrics.json")))
    upload_dir: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_UPLOAD_DIR", "storage/uploads")))
    prediction_limit_default: int = int(os.getenv("HOTEL_RISK_PREDICTION_LIMIT", "500"))
    sync_max_rows: int = int(os.getenv("HOTEL_RISK_SYNC_MAX_ROWS", "1000000"))
    max_upload_bytes: int = int(os.getenv("HOTEL_RISK_MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
    require_trained_model: bool = Field(default_factory=lambda: _bool_env("HOTEL_RISK_REQUIRE_TRAINED_MODEL", "0"))
    allow_fallback_model: bool = Field(default_factory=lambda: _bool_env("HOTEL_RISK_ALLOW_FALLBACK_MODEL", "1"))
    cors_origins: list[str] = Field(default_factory=lambda: _csv_env("HOTEL_RISK_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"))
    random_state: int = int(os.getenv("HOTEL_RISK_RANDOM_STATE", "42"))
    log_level: str = os.getenv("HOTEL_RISK_LOG_LEVEL", "INFO")
    log_file: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_LOG_FILE", "storage/logs/app.log")))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    return settings
