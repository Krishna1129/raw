"""FastAPI contract and lifecycle tests."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from src.explainability import ExplainabilityError
from src.preprocessing import get_model_features, load_raw_dataset


def _payload() -> dict[str, object]:
    row = get_model_features(load_raw_dataset()).iloc[0].to_dict()
    return {key: (None if value != value else value) for key, value in row.items()}


def test_health_and_ready_load_bundle_once() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"
        response = client.post("/predict", json=_payload())
        assert response.status_code == 200
        result = response.json()
        assert result["delay_risk_level"] in {"Low", "Medium", "High"}
        assert 0 <= result["risk_probability"] <= 1
        assert len(result["top_risk_factors"]) <= 3
        assert result["model_version"] == "1.0.0"


def test_extra_target_field_is_rejected() -> None:
    payload = _payload()
    payload["risk_probability"] = 0.5
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"
    assert response.json()["request_id"]


def test_invalid_percentage_is_rejected() -> None:
    payload = _payload()
    payload["private_land_pct"] = 101
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_unknown_categories_are_handled() -> None:
    payload = _payload()
    payload["state"] = "Previously unseen state"
    payload["project_type"] = "Previously unseen project"
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_prediction_errors_do_not_expose_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("secret C:\\local\\model path and stack detail")

    monkeypatch.setattr(api_main, "build_classifier_explainer", fail)
    with pytest.raises(RuntimeError):
        with TestClient(app):
            pass


def test_model_bundle_is_loaded_once_per_app_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = api_main.load_model_bundle
    calls = 0

    def counted(path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(api_main, "load_model_bundle", counted)
    with TestClient(app) as client:
        client.get("/ready")
        client.get("/ready")
        client.post("/predict", json=_payload())
    assert calls == 1


def test_explanation_failure_is_controlled_and_does_not_leak_details() -> None:
    with TestClient(app, raise_server_exceptions=False) as client:
        explainer = client.app.state.explainer

        def fail(*args: object, **kwargs: object) -> object:
            raise ExplainabilityError("internal C:\\secret\\path")

        explainer.explain = fail
        response = client.post("/predict", json=_payload())
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "prediction_failed"
    assert "secret" not in response.text
    assert "Traceback" not in response.text
    assert "C:\\" not in response.text


def test_missing_model_artifact_prevents_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api_main.settings, "model_artifact_path", tmp_path / "missing.pkl")
    with pytest.raises(RuntimeError, match="does not exist"):
        with TestClient(app):
            pass
