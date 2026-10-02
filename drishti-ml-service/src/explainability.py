"""SHAP explanations for the fitted classifier pipeline.

Attributions are model-specific contributions in transformed feature space.
They are not percentages, proof of causation, or evidence that a feature caused
the underlying project outcome.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import shap

from src.preprocessing import validate_inference_features

logger = logging.getLogger(__name__)


class ExplainabilityError(RuntimeError):
    """Raised when a reliable explanation cannot be produced."""


def normalize_shap_values(
    values: Any,
    *,
    n_samples: int,
    n_features: int,
    class_index: int,
) -> np.ndarray:
    """Normalize SHAP's list, 2-D, and 3-D multiclass outputs to one 2-D array."""
    arrays = values if isinstance(values, list) else None
    if arrays is not None:
        if not arrays or class_index < 0 or class_index >= len(arrays):
            raise ExplainabilityError(f"Invalid SHAP class index {class_index} for list output")
        selected = np.asarray(arrays[class_index])
        if selected.shape != (n_samples, n_features):
            raise ExplainabilityError(
                f"Unsupported SHAP list item shape {selected.shape}; "
                f"expected {(n_samples, n_features)}"
            )
        return selected

    array = np.asarray(values)
    if array.ndim == 2:
        if array.shape != (n_samples, n_features):
            raise ExplainabilityError(
                f"Unsupported SHAP 2-D shape {array.shape}; expected {(n_samples, n_features)}"
            )
        return array
    if array.ndim != 3:
        raise ExplainabilityError(f"Unsupported SHAP output dimensions: {array.shape}")

    candidates: list[np.ndarray] = []
    if array.shape == (n_samples, n_features, array.shape[2]):
        if class_index < array.shape[2]:
            candidates.append(array[:, :, class_index])
    if array.shape[0] > class_index and array.shape[1:] == (n_samples, n_features):
        candidates.append(array[class_index, :, :])
    if array.shape[1] > class_index and (array.shape[0], array.shape[2]) == (
        n_samples,
        n_features,
    ):
        candidates.append(array[:, class_index, :])
    if len(candidates) != 1:
        raise ExplainabilityError(
            f"Ambiguous or unsupported SHAP 3-D shape {array.shape} for "
            f"{n_samples} samples, {n_features} features"
        )
    return candidates[0]


def _tree_estimator(pipeline: Any) -> Any:
    try:
        estimator = pipeline.named_steps["model"]
    except (AttributeError, KeyError) as exc:
        raise ExplainabilityError("Classifier pipeline must contain a 'model' step") from exc
    # The custom XGBoost adapter stores its fitted model as ``estimator_``.
    # sklearn forests also expose an unfitted template named ``estimator_``;
    # those must remain intact for TreeExplainer.
    if hasattr(estimator, "estimators_"):
        return estimator
    return getattr(estimator, "estimator_", estimator)


def _feature_display_name(feature: str, mapping: Mapping[str, str]) -> str:
    if feature in mapping:
        return mapping[feature]
    _, _, feature = feature.partition("__")
    source, separator, category = feature.partition("_")
    if separator and category and source in mapping:
        return f"{mapping[source]}: {category}"
    return feature.replace("_", " ").replace(":", " - ").title()


@dataclass
class ClassifierExplainer:
    """Reusable explainer bound to one fitted classification pipeline."""

    pipeline: Any
    class_labels: Sequence[str]
    display_names: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        try:
            self.preprocessor = self.pipeline.named_steps["preprocessor"]
        except (AttributeError, KeyError) as exc:
            raise ExplainabilityError("Classifier pipeline must contain a fitted preprocessor") from exc
        self.estimator = _tree_estimator(self.pipeline)
        if not hasattr(self.preprocessor, "get_feature_names_out"):
            raise ExplainabilityError("Fitted preprocessor cannot provide transformed feature names")
        self.feature_names = list(self.preprocessor.get_feature_names_out())
        try:
            self.explainer = shap.TreeExplainer(self.estimator)
        except Exception as exc:
            logger.exception("Unable to initialize SHAP TreeExplainer")
            raise ExplainabilityError(f"Selected classifier is not SHAP TreeExplainer compatible: {exc}") from exc
        self.display_names = self.display_names or {}

    def explain(self, raw_features: pd.DataFrame) -> dict[str, Any]:
        """Return the top three signed attributions for one raw feature row."""
        if len(raw_features) != 1:
            raise ValueError("Explainability currently requires exactly one input row")
        expected_columns = list(self.pipeline.feature_names_in_)
        ordered = validate_inference_features(raw_features, expected_columns)
        transformed = self.preprocessor.transform(ordered)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed)
        if transformed.shape[1] != len(self.feature_names):
            raise ExplainabilityError(
                f"Transformed feature count {transformed.shape[1]} does not match "
                f"name count {len(self.feature_names)}"
            )
        predicted = self.pipeline.predict(ordered)
        predicted_label = str(predicted[0])
        if predicted_label not in self.class_labels:
            raise ExplainabilityError(f"Predicted class {predicted_label!r} is not in class labels")
        estimator_labels = [str(label) for label in getattr(self.estimator, "classes_", self.class_labels)]
        if predicted_label not in estimator_labels:
            raise ExplainabilityError(
                f"Predicted class {predicted_label!r} is not in estimator classes {estimator_labels}"
            )
        class_index = estimator_labels.index(predicted_label)
        try:
            raw_values = self.explainer.shap_values(transformed)
            values = normalize_shap_values(
                raw_values,
                n_samples=1,
                n_features=len(self.feature_names),
                class_index=class_index,
            )[0]
        except ExplainabilityError:
            raise
        except Exception as exc:
            logger.exception("SHAP explanation failed")
            raise ExplainabilityError(f"Unable to calculate SHAP values: {exc}") from exc
        if not np.isfinite(values).all():
            raise ExplainabilityError("SHAP values contain NaN or infinity")
        top_indices = np.argsort(np.abs(values))[::-1][:3]
        factors = []
        for index in top_indices:
            contribution = float(values[index])
            factors.append(
                {
                    "feature": str(self.feature_names[index]),
                    "contribution": contribution,
                    "direction": (
                        "increases_predicted_class"
                        if contribution >= 0
                        else "decreases_predicted_class"
                    ),
                    "display_name": _feature_display_name(
                        str(self.feature_names[index]), self.display_names
                    ),
                    "description": (
                        f"Model-attributed contribution toward the {predicted_label} "
                        "classification; not a causal effect."
                    ),
                }
            )
        return {
            "predicted_class": predicted_label,
            "factors": factors,
            "attribution_note": (
                "SHAP values are model-specific contributions in transformed feature space; "
                "they are not percentages or proof of causation."
            ),
        }


def build_classifier_explainer(
    bundle: Mapping[str, Any],
    display_names: Mapping[str, str] | None = None,
) -> ClassifierExplainer:
    """Create one reusable classifier explainer when a trusted bundle is loaded."""
    return ClassifierExplainer(
        pipeline=bundle["classification_pipeline"],
        class_labels=bundle["class_labels"],
        display_names=display_names,
    )
