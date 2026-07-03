from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ml import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MVP scoring model.")
    parser.add_argument("--data", default="data/processed/main_modeling_dataset.csv")
    parser.add_argument("--model", default="models/hw8_model.joblib")
    parser.add_argument("--metrics", default="reports/tables/hw8_model_metrics.json")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    metrics = train_model(Path(args.data), Path(args.model), Path(args.metrics), args.max_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
