"""FastAPI inference service for the DRISHTI risk models."""

from __future__ import annotations

import logging
import math
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_bundle, get_explainer, prepare_inference_features
from api.exception_handlers import (
    internal_exception_handler,
    prediction_exception_handler,
    validation_exception_handler,
)
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    ReadyResponse,
)
from src.config import settings
from src.explainability import ExplainabilityError, build_classifier_explainer
from src.model_bundle import load_model_bundle

logger = logging.getLogger(__name__)


def _validate_bundle(bundle: dict[str, Any]) -> None:
    required = {
        "artifact_version",
        "classification_pipeline",
        "regression_pipeline",
        "feature_columns",
        "numeric_features",
        "categorical_features",
        "class_labels",
    }
    missing = required - set(bundle)
    if missing:
        raise RuntimeError(f"Model bundle is missing required keys: {sorted(missing)}")
    if bundle["class_labels"] != ["Low", "Medium", "High"]:
        raise RuntimeError("Model bundle class labels are incompatible with the API")
    if not bundle["feature_columns"]:
        raise RuntimeError("Model bundle contains no feature metadata")
    for name in ("classification_pipeline", "regression_pipeline"):
        pipeline = bundle[name]
        if not hasattr(pipeline, "predict") or not hasattr(pipeline, "named_steps"):
            raise RuntimeError(f"{name} is not a fitted sklearn pipeline")


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact_path = settings.model_artifact_path
    if not artifact_path.is_file():
        raise RuntimeError(f"Configured model artifact does not exist: {artifact_path}")
    bundle = load_model_bundle(artifact_path)
    _validate_bundle(bundle)
    explainer = build_classifier_explainer(bundle)
    app.state.model_bundle = bundle
    app.state.explainer = explainer
    app.state.ready = True
    logger.info("Loaded model bundle version=%s", bundle["artifact_version"])
    try:
        yield
    finally:
        app.state.ready = False
        app.state.explainer = None
        app.state.model_bundle = None


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "DRISHTI land-acquisition risk inference API. `risk_probability` is a "
        "continuous regressor score, not the classifier's probability of the High class. "
        "Models were trained on synthetic data and outputs require domain validation."
    ),
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    return await call_next(request)


app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ExplainabilityError, prediction_exception_handler)
app.add_exception_handler(ValueError, prediction_exception_handler)
app.add_exception_handler(Exception, internal_exception_handler)


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["operations"])
def ready(request: Request) -> ReadyResponse:
    bundle = get_bundle(request)
    if not getattr(request.app.state, "ready", False):
        raise RuntimeError("Service is not ready")
    return ReadyResponse(status="ready", model_version=str(bundle["artifact_version"]))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="Predict delay risk and return top model-attributed factors",
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    bundle = get_bundle(request)
    explainer = get_explainer(request)
    raw_features = prepare_inference_features(payload.model_dump(), bundle["feature_columns"])
    class_pipeline = bundle["classification_pipeline"]
    regression_pipeline = bundle["regression_pipeline"]
    predicted_class = str(class_pipeline.predict(raw_features)[0])
    if predicted_class not in {"Low", "Medium", "High"}:
        raise ValueError("Classifier returned an unsupported risk category")
    score = float(np.clip(regression_pipeline.predict(raw_features)[0], 0.0, 1.0))
    if not math.isfinite(score):
        raise ValueError("Regression model returned a non-finite score")
    explanation = explainer.explain(raw_features)
    response = PredictionResponse(
        delay_risk_level=predicted_class,
        risk_probability=score,
        top_risk_factors=explanation["factors"],
        model_version=str(bundle["artifact_version"]),
    )
    return response
