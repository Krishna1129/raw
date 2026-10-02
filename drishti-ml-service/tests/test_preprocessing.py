"""Tests for the canonical preprocessing system."""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import (
    CLASS_TARGET,
    LEAKAGE_COLUMNS,
    DatasetValidationError,
    build_preprocessor,
    get_model_features,
    load_and_validate_dataset,
    separate_targets,
    split_dataset,
    validate_inference_features,
)

DATASET = Path(__file__).parents[1] / "data" / "raw" / "drishti_land_acquisition_dataset.csv"


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return load_and_validate_dataset(DATASET)


def test_targets_and_leakage_are_separated(dataset: pd.DataFrame) -> None:
    X, y_class, y_regression = separate_targets(dataset)
    assert len(X) == len(y_class) == len(y_regression)
    assert y_class.name == CLASS_TARGET
    assert y_regression.name == "risk_probability"
    assert not set(X.columns) & LEAKAGE_COLUMNS


def test_split_dimensions_and_reproducibility(dataset: pd.DataFrame) -> None:
    first = split_dataset(dataset)
    second = split_dataset(dataset)
    assert len(first.X_train) == 20_000
    assert len(first.X_test) == 5_000
    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_regression_test, second.y_regression_test)
    assert set(first.X_train.index).isdisjoint(first.X_test.index)


def test_missing_values_and_unseen_categories_are_supported(dataset: pd.DataFrame) -> None:
    X = get_model_features(dataset).iloc[:100].copy()
    numeric_column = X.select_dtypes(include="number").columns[0]
    categorical_column = X.select_dtypes(exclude="number").columns[0]
    X.loc[X.index[0], numeric_column] = None
    X.loc[X.index[1], categorical_column] = None
    preprocessor = build_preprocessor(X)
    preprocessor.fit(X)
    inference = X.iloc[:2].copy()
    inference[categorical_column] = "unseen-category"
    transformed = preprocessor.transform(validate_inference_features(inference, X.columns))
    assert transformed.shape[0] == 2


def test_inference_feature_order_is_canonical(dataset: pd.DataFrame) -> None:
    X = get_model_features(dataset).iloc[:3]
    ordered = validate_inference_features(X[X.columns[::-1]], X.columns)
    assert ordered.columns.tolist() == X.columns.tolist()


def test_missing_required_inference_feature_fails(dataset: pd.DataFrame) -> None:
    X = get_model_features(dataset).iloc[:1]
    with pytest.raises(ValueError, match="Missing required feature"):
        validate_inference_features(X.drop(columns=[X.columns[0]]), X.columns)


def test_invalid_probability_fails(tmp_path: Path, dataset: pd.DataFrame) -> None:
    invalid = dataset.copy()
    invalid.loc[0, "risk_probability"] = 1.5
    path = tmp_path / "invalid.csv"
    invalid.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="between 0.0 and 1.0"):
        load_and_validate_dataset(path)


def test_missing_dataset_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="does not exist"):
        load_and_validate_dataset(tmp_path / "missing.csv")


def test_missing_target_columns_fail_clearly(tmp_path: Path, dataset: pd.DataFrame) -> None:
    path = tmp_path / "missing-target.csv"
    dataset.drop(columns=["delay_risk_level"]).to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="required target columns"):
        load_and_validate_dataset(path)


def test_duplicate_project_ids_fail_clearly(tmp_path: Path, dataset: pd.DataFrame) -> None:
    duplicate = dataset.iloc[:2].copy()
    duplicate.loc[duplicate.index[1], "project_id"] = duplicate.iloc[0]["project_id"]
    path = tmp_path / "duplicate-id.csv"
    duplicate.to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="project_id must be unique"):
        load_and_validate_dataset(path)
