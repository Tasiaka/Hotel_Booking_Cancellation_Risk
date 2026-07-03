from __future__ import annotations

import os

from .config import get_settings
from .db import init_db


def main() -> None:
    try:
        from redis import Redis
        from rq import Worker, Queue
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Install redis and rq to run model workers: pip install redis rq") from exc
    settings = get_settings()
    init_db()
    redis = Redis.from_url(settings.redis_url)
    queues = [Queue("hotel-risk-scoring", connection=redis)]
    name = os.getenv("WORKER_NAME", None)
    worker = Worker(queues, connection=redis, name=name)
    worker.work(with_scheduler=False)


if __name__ == "__main__":
    main()
