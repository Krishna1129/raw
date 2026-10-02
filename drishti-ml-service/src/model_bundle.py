"""Trusted Joblib serialization for complete inference-ready model bundles."""

from __future__ import annotations

import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas
import sklearn
import numpy as np
import xgboost
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from xgboost import XGBClassifier


class EncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier that maps integer labels back to domain labels."""

    def __init__(self, *, random_state: int = 42, n_estimators: int = 80, n_jobs: int = -1) -> None:
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> "EncodedXGBClassifier":
        self.classes_ = np.asarray(["Low", "Medium", "High"])
        mapping = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([mapping[str(value)] for value in y], dtype=int)
        self.estimator_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
        )
        self.estimator_.fit(X, encoded)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.estimator_.predict(X).astype(int)]

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Regressor adapter that enforces the risk score's [0, 1] output range."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Any) -> "ClippedRegressor":
        self.estimator_ = self.estimator.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(np.asarray(self.estimator_.predict(X), dtype=float), 0.0, 1.0)


def create_model_bundle(
    *,
    classification_pipeline: Any,
    regression_pipeline: Any,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    class_labels: list[str],
    selected_classification_model: str,
    selected_regression_model: str,
    classification_metrics: dict[str, Any],
    regression_metrics: dict[str, Any],
    random_seed: int,
    dataset_schema_version: str = "phase-2-v1",
) -> dict[str, Any]:
    return {
        "artifact_version": "1.0.0",
        "classification_pipeline": classification_pipeline,
        "regression_pipeline": regression_pipeline,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "class_labels": class_labels,
        "selected_classification_model": selected_classification_model,
        "selected_regression_model": selected_regression_model,
        "classification_metrics": classification_metrics,
        "regression_metrics": regression_metrics,
        "random_seed": random_seed,
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "pandas_version": pandas.__version__,
        "numpy_version": numpy.__version__,
        "python_version": platform.python_version(),
        "dataset_schema_version": dataset_schema_version,
    }


def save_model_bundle(bundle: dict[str, Any], artifact_path: Path) -> Path:
    """Atomically save a trusted internal artifact with Joblib."""
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_path.name}.", suffix=".tmp", dir=artifact_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        joblib.dump(bundle, temporary_path)
        temporary_path.replace(artifact_path)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to serialize model bundle to {artifact_path}: {exc}") from exc
    return artifact_path


def load_model_bundle(artifact_path: Path) -> dict[str, Any]:
    """Load a trusted internal Joblib artifact; never use this for untrusted files."""
    try:
        bundle = joblib.load(Path(artifact_path))
    except Exception as exc:
        raise RuntimeError(f"Unable to load trusted model bundle {artifact_path}: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("artifact_version") is None:
        raise ValueError("Invalid model bundle: expected a versioned dictionary")
    return bundle
