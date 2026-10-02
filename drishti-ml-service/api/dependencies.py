"""Dependencies and inference helpers shared by API routes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import Request

from src.explainability import ClassifierExplainer
from src.preprocessing import validate_inference_features


def get_bundle(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise RuntimeError("Model bundle is not loaded")
    return bundle


def get_explainer(request: Request) -> ClassifierExplainer:
    explainer = getattr(request.app.state, "explainer", None)
    if explainer is None:
        raise RuntimeError("SHAP explainer is not initialized")
    return explainer


def prepare_inference_features(payload: dict[str, Any], expected_columns: list[str]) -> pd.DataFrame:
    """Convert a validated request to the exact fitted feature order."""
    frame = pd.DataFrame([payload])
    return validate_inference_features(frame, expected_columns)
