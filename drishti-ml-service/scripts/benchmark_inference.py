"""Benchmark the loaded inference path and report latency percentiles.

This benchmark uses the production bundle and FastAPI TestClient. It measures
warm requests only; model loading and SHAP explainer construction are excluded
from request latency. The HTTP measurement includes JSON serialization and the
in-process ASGI round trip, not network latency to a separately hosted server.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# Make direct ``python scripts\benchmark_inference.py`` execution equivalent
# to running the module from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.dependencies import prepare_inference_features
from api.main import app
from src.config import settings
from src.model_bundle import load_model_bundle
from src.preprocessing import get_model_features, load_raw_dataset


def _milliseconds(duration_seconds: float) -> float:
    return duration_seconds * 1000.0


def _summary(samples_ms: list[float], *, includes_shap: bool) -> dict[str, Any]:
    ordered = sorted(samples_ms)
    return {
        "measured_requests": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p95_ms": float(np.percentile(ordered, 95)),
        "p99_ms": float(np.percentile(ordered, 99)),
        "max_ms": max(ordered),
        "includes_shap": includes_shap,
    }


def _measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    requests: int,
    includes_shap: bool,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(requests):
        started = time.perf_counter()
        operation()
        samples.append(_milliseconds(time.perf_counter() - started))
    return _summary(samples, includes_shap=includes_shap)


def _native_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        payload[key] = value.item() if isinstance(value, np.generic) else value
    return payload


def run_benchmark(*, requests: int = 100, warmup: int = 10) -> dict[str, Any]:
    """Run separate warm-up/repeated measurements for every inference stage."""
    bundle = load_model_bundle(settings.model_artifact_path)
    frame = load_raw_dataset()
    raw_features = get_model_features(frame).iloc[[0]]
    payload = _native_payload(raw_features.iloc[0])

    with TestClient(app) as client:
        client.get("/ready")
        loaded_bundle = client.app.state.model_bundle
        explainer = client.app.state.explainer
        feature_columns = loaded_bundle["feature_columns"]

        def prepared() -> pd.DataFrame:
            return prepare_inference_features(payload, feature_columns)

        def classification() -> Any:
            return loaded_bundle["classification_pipeline"].predict(prepared())

        def regression() -> Any:
            return loaded_bundle["regression_pipeline"].predict(prepared())

        def shap_explanation() -> Any:
            return explainer.explain(prepared())

        def complete_processing() -> Any:
            features = prepared()
            loaded_bundle["classification_pipeline"].predict(features)
            loaded_bundle["regression_pipeline"].predict(features)
            return explainer.explain(features)

        def http_round_trip() -> Any:
            response = client.post("/predict", json=payload)
            response.raise_for_status()
            return response.json()

        results = {
            "model_only_classification": _measure(
                classification, warmup=warmup, requests=requests, includes_shap=False
            ),
            "model_only_regression": _measure(
                regression, warmup=warmup, requests=requests, includes_shap=False
            ),
            "shap_explanation": _measure(
                shap_explanation, warmup=warmup, requests=requests, includes_shap=True
            ),
            "complete_prediction_processing": _measure(
                complete_processing, warmup=warmup, requests=requests, includes_shap=True
            ),
            "http_round_trip": _measure(
                http_round_trip, warmup=warmup, requests=requests, includes_shap=True
            ),
        }

    complete_p95 = results["complete_prediction_processing"]["p95_ms"]
    http_p95 = results["http_round_trip"]["p95_ms"]
    return {
        "benchmark": {
            "target_ms": 120,
            "requests": requests,
            "warmup_requests": warmup,
            "latency_unit": "ms",
            "target_interpretation": "Measured objective, not a guarantee",
            "http_scope": "In-process FastAPI TestClient round trip; excludes external network hop",
            "shap_policy": "Included in complete processing and HTTP measurements",
            "complete_processing_p95_under_target": complete_p95 < 120,
            "http_round_trip_p95_under_target": http_p95 < 120,
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "model_artifact": str(Path(settings.model_artifact_path)),
            "artifact_version": str(bundle["artifact_version"]),
        },
        "measurements": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100, help="Measured requests per operation")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up requests per operation")
    args = parser.parse_args()
    if args.requests < 2 or args.warmup < 0:
        parser.error("--requests must be at least 2 and --warmup cannot be negative")
    print(json.dumps(run_benchmark(requests=args.requests, warmup=args.warmup), indent=2))


if __name__ == "__main__":
    main()
