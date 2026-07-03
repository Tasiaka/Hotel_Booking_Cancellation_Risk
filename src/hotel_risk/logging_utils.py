from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(name: str = "hotel_risk") -> logging.Logger:
    """Configure console + rotating file logging once per process.

    Env vars:
    - HOTEL_RISK_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
    - HOTEL_RISK_LOG_FILE=storage/logs/app.log
    """
    global _CONFIGURED
    level_name = os.getenv("HOTEL_RISK_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = Path(os.getenv("HOTEL_RISK_LOG_FILE", "storage/logs/app.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("hotel_risk")
    root.setLevel(level)

    if not _CONFIGURED:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(formatter)
        root.addHandler(console)

        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        # keep third-party noise lower unless explicitly requested
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("multipart").setLevel(logging.WARNING)
        _CONFIGURED = True

    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
