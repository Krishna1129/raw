"""End-to-end training and model bundle tests."""

from pathlib import Path

import pandas as pd

from src.model_bundle import load_model_bundle
from src.preprocessing import get_model_features, load_and_validate_dataset
from src.train import train_and_save

DATASET = Path(__file__).parents[1] / "data" / "raw" / "drishti_land_acquisition_dataset.csv"


def test_training_serializes_complete_inference_bundle(tmp_path: Path) -> None:
    source = load_and_validate_dataset(DATASET)
    fixture = pd.concat(
        [
            source[source["delay_risk_level"] == label].iloc[:30]
            for label in ("Low", "Medium", "High")
        ],
        ignore_index=True,
    )
    fixture_path = tmp_path / "fixture.csv"
    fixture.to_csv(fixture_path, index=False)
    result = train_and_save(
        data_path=fixture_path,
        artifact_path=tmp_path / "pipeline_with_preprocessor.pkl",
        reports_directory=tmp_path / "reports",
    )
    bundle = load_model_bundle(result.artifact_path)
    assert bundle["selected_classification_model"] in {"random_forest", "xgboost"}
    assert bundle["selected_regression_model"] in {"random_forest", "xgboost"}
    assert bundle["feature_columns"] == get_model_features(fixture).columns.tolist()
    assert set(bundle["class_labels"]) == {"Low", "Medium", "High"}
    raw_features = get_model_features(fixture).iloc[:5]
    class_predictions = bundle["classification_pipeline"].predict(raw_features)
    regression_predictions = bundle["regression_pipeline"].predict(raw_features)
    assert class_predictions.shape == (5,)
    assert regression_predictions.shape == (5,)
    assert set(class_predictions).issubset({"Low", "Medium", "High"})
    assert ((regression_predictions >= 0.0) & (regression_predictions <= 1.0)).all()
    assert all(pd.notna(value) for value in regression_predictions)
    assert {"accuracy", "f1_macro", "confusion_matrix"}.issubset(
        bundle["classification_metrics"]
    )
    assert {"mae", "rmse", "r2"}.issubset(bundle["regression_metrics"])
    assert bundle["classification_metrics"]["classification_report"]["macro avg"]["support"] == 18
    assert bundle["regression_metrics"]["mae"] >= 0
    assert (tmp_path / "reports" / "classification_metrics.json").is_file()
    assert (tmp_path / "reports" / "regression_metrics.json").is_file()
