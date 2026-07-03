from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None
    from sklearn.ensemble import HistGradientBoostingClassifier

from .business import enrich_predictions
from .config import get_settings
from .features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET_ENCODING_OUTPUT_FEATURES, model_matrix, prepare_features


class RuleFallbackModel:
    """Deterministic fallback used when no trained artifact is available."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        score = np.zeros(len(x), dtype=float) + 0.12
        score += (pd.to_numeric(x.get("lead_time", 0), errors="coerce").fillna(0).to_numpy() > 90) * 0.18
        score += (pd.to_numeric(x.get("lead_time", 0), errors="coerce").fillna(0).to_numpy() > 180) * 0.12
        score += (x.get("deposit_type", "").astype(str).to_numpy() == "No Deposit") * 0.10
        score += (x.get("market_segment", "").astype(str).isin(["Online TA", "Groups"]).to_numpy()) * 0.15
        score += (x.get("distribution_channel", "").astype(str).to_numpy() == "TA/TO") * 0.08
        score += (pd.to_numeric(x.get("previous_cancellations", 0), errors="coerce").fillna(0).to_numpy() > 0) * 0.22
        score += (pd.to_numeric(x.get("total_of_special_requests", 0), errors="coerce").fillna(0).to_numpy() == 0) * 0.08
        score -= (pd.to_numeric(x.get("required_car_parking_spaces", 0), errors="coerce").fillna(0).to_numpy() > 0) * 0.20
        score -= (pd.to_numeric(x.get("is_repeated_guest", 0), errors="coerce").fillna(0).to_numpy() == 1) * 0.15
        score = np.clip(score, 0.02, 0.98)
        return np.vstack([1 - score, score]).T


class TrainOnlyTargetFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Add leakage-safe frequency and target-mean features computed only on train.

    The transformer receives prepared booking features and the training target.
    It stores category statistics from train only, then applies them to validation,
    test, API and worker inference. Unknown categories fall back to the train prior.
    """

    def __init__(self, smoothing: float = 60.0):
        self.smoothing = smoothing
        self.columns = [
            "agent",
            "company_raw",
            "country",
            "market_segment",
            "distribution_channel",
            "deposit_type",
        ]

    @staticmethod
    def _as_key(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray | None = None):
        self.global_prior_ = float(np.mean(y)) if y is not None and len(y) else 0.35
        self.stats_: dict[str, dict[str, Any]] = {}
        y_series = pd.Series(y, index=X.index).astype(float) if y is not None else pd.Series(self.global_prior_, index=X.index)
        for col in self.columns:
            keys = self._as_key(X[col]) if col in X.columns else pd.Series("Unknown", index=X.index, dtype="string")
            tmp = pd.DataFrame({"key": keys, "target": y_series})
            grp = tmp.groupby("key", dropna=False)["target"].agg(["count", "mean"])
            count = grp["count"].astype(float)
            mean = grp["mean"].astype(float)
            smooth_mean = (mean * count + self.global_prior_ * self.smoothing) / (count + self.smoothing)
            self.stats_[col] = {
                "count": count.to_dict(),
                "smooth_mean": smooth_mean.to_dict(),
            }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            keys = self._as_key(out[col]) if col in out.columns else pd.Series("Unknown", index=out.index, dtype="string")
            stats = getattr(self, "stats_", {}).get(col, {"count": {}, "smooth_mean": {}})
            counts = keys.map(stats.get("count", {})).fillna(0).astype(float)
            means = keys.map(stats.get("smooth_mean", {})).fillna(getattr(self, "global_prior_", 0.35)).astype(float)
            prefix = "company" if col == "company_raw" else col
            if col in {"agent", "company_raw"}:
                out[f"{prefix}_freq_log"] = np.log1p(counts)
                out[f"{prefix}_target_mean_smooth"] = means
            else:
                out[f"{prefix}_target_mean_smooth"] = means
        for feature in TARGET_ENCODING_OUTPUT_FEATURES:
            if feature not in out.columns:
                out[feature] = getattr(self, "global_prior_", 0.35) if feature.endswith("target_mean_smooth") else 0.0
        return out


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True)
    except TypeError:  # sklearn<1.2
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse=True)


def build_pipeline(random_state: int = 42, params: dict[str, Any] | None = None) -> Pipeline:
    params = params or {}
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _one_hot_encoder()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    if LGBMClassifier is not None:
        default_params = dict(
            n_estimators=260,
            learning_rate=0.05,
            num_leaves=64,
            max_depth=7,
            min_child_samples=70,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            reg_alpha=0.5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            force_row_wise=True,
            verbose=-1,
        )
        default_params.update(params)
        clf = LGBMClassifier(**default_params)
    else:  # pragma: no cover
        clf = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.05, random_state=random_state)
    return Pipeline([
        ("target_frequency_encoder", TrainOnlyTargetFrequencyEncoder(smoothing=60.0)),
        ("preprocess", pre),
        ("model", clf),
    ])


def model_candidates(random_state: int) -> list[tuple[str, Pipeline]]:
    if LGBMClassifier is None:  # pragma: no cover
        return [("HistGradientBoosting fallback", build_pipeline(random_state))]
    configs: list[tuple[str, dict[str, Any]]] = [
        (
            "LightGBM FE TE conservative",
            dict(n_estimators=260, learning_rate=0.05, num_leaves=64, max_depth=7, min_child_samples=70, reg_lambda=10.0, reg_alpha=0.5),
        ),
        (
            "LightGBM FE TE stronger",
            dict(n_estimators=340, learning_rate=0.04, num_leaves=96, max_depth=8, min_child_samples=45, reg_lambda=7.0, reg_alpha=0.3),
        ),
    ]
    return [(name, build_pipeline(random_state, params)) for name, params in configs]


def fit_probability_calibrator(raw_scores: np.ndarray, y_true: pd.Series | np.ndarray) -> LogisticRegression | None:
    """Fit Platt calibration on validation raw probabilities.

    The calibrator is monotonic, so it keeps the ranking almost unchanged,
    but maps raw scores to probabilities suitable for expected-loss estimates.
    """
    y = np.asarray(y_true).astype(int)
    raw = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(raw, y)
    return calibrator


def apply_probability_calibration(raw_scores: np.ndarray, calibrator: Any | None = None) -> np.ndarray:
    raw = np.clip(np.asarray(raw_scores, dtype=float), 0.0, 1.0)
    if calibrator is None:
        return raw
    try:
        return np.clip(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], 0.0, 1.0)
    except Exception:
        return raw


def topk_metrics(y_true: pd.Series | np.ndarray, score: np.ndarray, k: float = 0.2) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    n = max(1, int(len(y) * k))
    order = np.argsort(score)[::-1][:n]
    precision = float(y[order].mean())
    recall = float(y[order].sum() / max(1, y.sum()))
    lift = float(precision / max(1e-9, y.mean()))
    return {f"precision@{int(k*100)}": precision, f"recall@{int(k*100)}": recall, f"lift@{int(k*100)}": lift}


def train_model(data_path: str | Path, model_path: str | Path | None = None, metrics_path: str | Path | None = None, max_rows: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    model_path = Path(model_path or settings.model_path)
    metrics_path = Path(metrics_path or settings.metrics_path)
    df = pd.read_csv(data_path)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(max_rows, random_state=settings.random_state).sort_index()
    if "split" not in df.columns:
        df = df.sort_values(["arrival_date_year", "arrival_date_week_number", "arrival_date_day_of_month"], kind="stable")
        cut1 = int(len(df) * 0.70)
        cut2 = int(len(df) * 0.85)
        df["split"] = "train"
        df.loc[df.index[cut1:cut2], "split"] = "valid"
        df.loc[df.index[cut2:], "split"] = "test"

    train = df[df["split"] == "train"].copy()
    valid = df[df["split"] == "valid"].copy()
    test = df[df["split"] == "test"].copy()
    if test.empty:
        test = df.sample(frac=0.2, random_state=settings.random_state)
        remaining = df.drop(index=test.index)
        valid = remaining.sample(frac=0.2, random_state=settings.random_state)
        train = remaining.drop(index=valid.index)
    if valid.empty:
        valid = train.sample(frac=0.2, random_state=settings.random_state)
        train = train.drop(index=valid.index)

    x_train, _, _ = model_matrix(train, strict=False)
    x_valid, _, _ = model_matrix(valid, strict=False)
    x_test, _, _ = model_matrix(test, strict=False)
    y_train = train["is_canceled"].astype(int)
    y_valid = valid["is_canceled"].astype(int)
    y_test = test["is_canceled"].astype(int)

    best_name = ""
    best_pipeline: Pipeline | None = None
    best_valid_raw: np.ndarray | None = None
    best_selection_score = -np.inf
    candidate_metrics: list[dict[str, Any]] = []
    for name, candidate in model_candidates(settings.random_state):
        candidate.fit(x_train, y_train)
        valid_raw_candidate = candidate.predict_proba(x_valid)[:, 1]
        valid_roc = float(roc_auc_score(y_valid, valid_raw_candidate))
        valid_pr = float(average_precision_score(y_valid, valid_raw_candidate))
        valid_lift = topk_metrics(y_valid, valid_raw_candidate, 0.2)["lift@20"]
        selection_score = 0.50 * valid_roc + 0.30 * valid_pr + 0.20 * valid_lift / 3.0
        candidate_metrics.append({
            "name": name,
            "valid_roc_auc": valid_roc,
            "valid_pr_auc": valid_pr,
            "valid_lift@20": valid_lift,
            "selection_score": float(selection_score),
        })
        if selection_score > best_selection_score:
            best_name = name
            best_pipeline = candidate
            best_valid_raw = valid_raw_candidate
            best_selection_score = float(selection_score)

    if best_pipeline is None or best_valid_raw is None:  # pragma: no cover
        raise RuntimeError("No model candidate was fitted")

    pipeline = best_pipeline
    valid_raw = best_valid_raw
    calibrator = fit_probability_calibrator(valid_raw, y_valid)

    test_raw = pipeline.predict_proba(x_test)[:, 1]
    test_calibrated = apply_probability_calibration(test_raw, calibrator)
    metrics: dict[str, Any] = {
        "model": best_name + " + Platt calibration",
        "model_selection": "best candidate by validation ROC-AUC/PR-AUC/Lift@20",
        "calibration_method": "Platt sigmoid on validation split" if calibrator is not None else "not fitted; raw score fallback",
        "expected_loss_probability": "cancellation_probability",
        "rows_train": int(len(train)),
        "rows_calibration": int(len(valid)),
        "rows_test": int(len(test)),
        "roc_auc": float(roc_auc_score(y_test, test_calibrated)),
        "pr_auc": float(average_precision_score(y_test, test_calibrated)),
        "brier": float(brier_score_loss(y_test, test_calibrated)),
        "roc_auc_raw": float(roc_auc_score(y_test, test_raw)),
        "pr_auc_raw": float(average_precision_score(y_test, test_raw)),
        "brier_raw": float(brier_score_loss(y_test, test_raw)),
        "candidate_metrics": candidate_metrics,
    }
    metrics.update(topk_metrics(y_test, test_raw, 0.2))
    artifact = {
        "pipeline": pipeline,
        "probability_calibrator": calibrator,
        "expected_loss_probability": "cancellation_probability",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "metrics": metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(model_path or get_settings().model_path)
    if path.exists():
        return joblib.load(path)
    return {"pipeline": RuleFallbackModel(), "metrics": {"model": "rule fallback; train with python -m hotel_risk.train"}}


def predict_dataframe(df: pd.DataFrame, model_artifact: dict[str, Any] | None = None, strict: bool = False) -> pd.DataFrame:
    x, prepared, _ = model_matrix(df, strict=strict)
    artifact = model_artifact or load_model()
    pipeline = artifact["pipeline"]
    raw_score = pipeline.predict_proba(x)[:, 1]
    calibrated_probability = apply_probability_calibration(raw_score, artifact.get("probability_calibrator"))
    return enrich_predictions(prepared, raw_score, calibrated_probability)
