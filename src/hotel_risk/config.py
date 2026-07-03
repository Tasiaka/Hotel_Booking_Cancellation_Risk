from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Hotel Booking Cancellation Risk"
    environment: str = Field(default_factory=lambda: os.getenv("HOTEL_RISK_ENV", "local"))
    database_url: str = Field(default_factory=lambda: os.getenv("HOTEL_RISK_DATABASE_URL", "sqlite:///./storage/hotel_risk.db"))
    redis_url: str = Field(default_factory=lambda: os.getenv("HOTEL_RISK_REDIS_URL", "redis://redis:6379/0"))
    model_path: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_MODEL_PATH", "models/hw8_model.joblib")))
    metrics_path: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_METRICS_PATH", "reports/tables/hw8_model_metrics.json")))
    upload_dir: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_UPLOAD_DIR", "storage/uploads")))
    prediction_limit_default: int = int(os.getenv("HOTEL_RISK_PREDICTION_LIMIT", "500"))
    sync_max_rows: int = int(os.getenv("HOTEL_RISK_SYNC_MAX_ROWS", "1000000"))
    random_state: int = int(os.getenv("HOTEL_RISK_RANDOM_STATE", "42"))
    log_level: str = os.getenv("HOTEL_RISK_LOG_LEVEL", "INFO")
    log_file: Path = Field(default_factory=lambda: Path(os.getenv("HOTEL_RISK_LOG_FILE", "storage/logs/app.log")))

    class Config:
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    return settings
