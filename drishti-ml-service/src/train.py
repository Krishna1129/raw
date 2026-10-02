"""Reproducible training, training-only selection, evaluation, and serialization."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import settings
from src.evaluation import evaluate_classifier, evaluate_regressor
from src.model_bundle import ClippedRegressor, EncodedXGBClassifier, create_model_bundle, save_model_bundle
from src.preprocessing import (
    CLASS_TARGET,
    REGRESSION_TARGET,
    build_preprocessor,
    get_model_features,
    load_and_validate_dataset,
    split_dataset,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    bundle: dict[str, Any]
    artifact_path: Path
    reports_directory: Path


def _classification_candidates(X: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=12,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [("preprocessor", build_preprocessor(X)), ("model", EncodedXGBClassifier(random_state=seed))]
        ),
    }


def _regression_candidates(X: pd.DataFrame, seed: int) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            [
                ("preprocessor", build_preprocessor(X)),
                (
                    "model",
                    ClippedRegressor(
                        RandomForestRegressor(
                            n_estimators=120,
                            max_depth=12,
                            random_state=seed,
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            [
                ("preprocessor", build_preprocessor(X)),
                (
                    "model",
                    ClippedRegressor(
                        XGBRegressor(
                            n_estimators=100,
                            max_depth=4,
                            learning_rate=0.08,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective="reg:squarederror",
                            random_state=seed,
                            n_jobs=-1,
                            tree_method="hist",
                        )
                    ),
                ),
            ]
        ),
    }


def _select_and_refit(
    candidates: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    classification: bool,
    seed: int,
) -> tuple[str, Pipeline]:
    stratify = y_train if classification else None
    X_fit, X_validate, y_fit, y_validate = train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=stratify
    )
    scores: dict[str, float] = {}
    for name, candidate in candidates.items():
        candidate.fit(X_fit, y_fit)
        predictions = candidate.predict(X_validate)
        scores[name] = (
            float(f1_score(y_validate, predictions, average="macro", zero_division=0))
            if classification
            else float(mean_absolute_error(y_validate, np.clip(predictions, 0.0, 1.0)))
        )
    selected = max(scores, key=scores.get) if classification else min(scores, key=scores.get)
    logger.info("Training-only %s selection scores=%s selected=%s", "classification" if classification else "regression", scores, selected)
    final = candidates[selected]
    final.fit(X_train, y_train)
    return selected, final


def train_and_save(
    data_path: Path = settings.dataset_path,
    artifact_path: Path = settings.model_artifact_path,
    reports_directory: Path = settings.reports_directory,
    random_seed: int = settings.random_seed,
) -> TrainingResult:
    """Run the complete reproducible training workflow."""
    frame = load_and_validate_dataset(data_path)
    splits = split_dataset(frame, random_state=random_seed)
    classification_name, classification_pipeline = _select_and_refit(
        _classification_candidates(splits.X_train, random_seed),
        splits.X_train,
        splits.y_class_train,
        classification=True,
        seed=random_seed,
    )
    regression_name, regression_pipeline = _select_and_refit(
        _regression_candidates(splits.X_train, random_seed),
        splits.X_train,
        splits.y_regression_train,
        classification=False,
        seed=random_seed,
    )
    class_labels = ["Low", "Medium", "High"]
    classification_metrics = evaluate_classifier(
        classification_pipeline, splits.X_test, splits.y_class_test, class_labels
    )
    regression_metrics = evaluate_regressor(
        regression_pipeline, splits.X_test, splits.y_regression_test
    )
    reports_directory = Path(reports_directory)
    reports_directory.mkdir(parents=True, exist_ok=True)
    (reports_directory / "classification_metrics.json").write_text(
        json.dumps(classification_metrics, indent=2), encoding="utf-8"
    )
    (reports_directory / "regression_metrics.json").write_text(
        json.dumps(regression_metrics, indent=2), encoding="utf-8"
    )
    X = get_model_features(frame)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    bundle = create_model_bundle(
        classification_pipeline=classification_pipeline,
        regression_pipeline=regression_pipeline,
        feature_columns=X.columns.tolist(),
        numeric_features=numeric,
        categorical_features=categorical,
        class_labels=class_labels,
        selected_classification_model=classification_name,
        selected_regression_model=regression_name,
        classification_metrics=classification_metrics,
        regression_metrics=regression_metrics,
        random_seed=random_seed,
    )
    saved_path = save_model_bundle(bundle, artifact_path)
    logger.info("Saved model bundle to %s", saved_path)
    return TrainingResult(bundle=bundle, artifact_path=saved_path, reports_directory=reports_directory)


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    result = train_and_save()
    print(json.dumps({"artifact_path": str(result.artifact_path), "classification": result.bundle["classification_metrics"], "regression": result.bundle["regression_metrics"]}, indent=2))
