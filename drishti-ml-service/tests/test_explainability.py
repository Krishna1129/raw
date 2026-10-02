"""Tests for SHAP explanation normalization and output safety."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.explainability import (
    ExplainabilityError,
    build_classifier_explainer,
    normalize_shap_values,
)
from src.model_bundle import load_model_bundle
from src.preprocessing import get_model_features, load_raw_dataset


def test_normalize_shap_output_variations() -> None:
    expected = np.arange(6, dtype=float).reshape(1, 6)
    assert np.array_equal(
        normalize_shap_values([expected, expected + 10], n_samples=1, n_features=6, class_index=1),
        expected + 10,
    )
    assert np.array_equal(
        normalize_shap_values(expected, n_samples=1, n_features=6, class_index=0), expected
    )
    three_d = np.stack([expected, expected + 10, expected + 20], axis=2)
    assert np.array_equal(
        normalize_shap_values(three_d, n_samples=1, n_features=6, class_index=2),
        expected + 20,
    )
    class_first = np.stack([expected, expected + 10, expected + 20], axis=0)
    assert np.array_equal(
        normalize_shap_values(class_first, n_samples=1, n_features=6, class_index=1),
        expected + 10,
    )


def test_unsupported_shap_shape_fails() -> None:
    with pytest.raises(ExplainabilityError, match="Unsupported SHAP"):
        normalize_shap_values(np.zeros((2, 3, 4, 5)), n_samples=1, n_features=3, class_index=0)


def test_production_explanation_is_safe_and_top_three() -> None:
    bundle_path = Path(__file__).parents[1] / "models" / "pipeline_with_preprocessor.pkl"
    bundle = load_model_bundle(bundle_path)
    explainer = build_classifier_explainer(bundle)
    features = get_model_features(load_raw_dataset()).iloc[[0]]
    explanation = explainer.explain(features)
    assert explanation["predicted_class"] in {"Low", "Medium", "High"}
    assert len(explanation["factors"]) <= 3
    assert len(explainer.feature_names) == len(
        explainer.preprocessor.get_feature_names_out()
    )
    for factor in explanation["factors"]:
        assert factor["feature"] in explainer.feature_names
        assert np.isfinite(factor["contribution"])
        assert factor["direction"] in {
            "increases_predicted_class",
            "decreases_predicted_class",
        }
    json.dumps(explanation)


def test_positive_and_negative_contributions_have_directions() -> None:
    values = np.array([[0.5, -0.25, 0.1]])
    normalized = normalize_shap_values(values, n_samples=1, n_features=3, class_index=0)[0]
    assert normalized[0] > 0
    assert normalized[1] < 0
