"""Metrics for the untouched held-out test partition."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def evaluate_classifier(
    pipeline: Any,
    X_test: Any,
    y_test: Sequence[str],
    class_labels: Sequence[str],
) -> dict[str, Any]:
    """Evaluate a classifier and explicitly represent unavailable ROC-AUC."""
    predictions = pipeline.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=list(class_labels),
        target_names=list(class_labels),
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(class_labels)).tolist(),
        "classification_report": report,
        "roc_auc_ovr": None,
    }
    try:
        probabilities = pipeline.predict_proba(X_test)
        if len(set(y_test)) == len(class_labels) and probabilities.shape[1] == len(class_labels):
            model_classes = getattr(pipeline, "classes_", class_labels)
            order = [list(model_classes).index(label) for label in class_labels]
            probabilities = probabilities[:, order]
            encoded_targets = label_binarize(y_test, classes=list(class_labels))
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(encoded_targets, probabilities, multi_class="ovr")
            )
    except (AttributeError, ValueError) as exc:
        metrics["roc_auc_error"] = str(exc)
    return metrics


def evaluate_regressor(pipeline: Any, X_test: Any, y_test: Sequence[float]) -> dict[str, float]:
    """Evaluate a continuous risk-score regressor on the same [0, 1] scale."""
    predictions = np.clip(np.asarray(pipeline.predict(X_test), dtype=float), 0.0, 1.0)
    target = np.asarray(y_test, dtype=float)
    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(target, predictions))),
        "r2": float(r2_score(target, predictions)),
        "spearman_correlation": float(
            np.corrcoef(np.argsort(np.argsort(target)), np.argsort(np.argsort(predictions)))[0, 1]
        ),
    }
