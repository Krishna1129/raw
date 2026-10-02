"""Canonical dataset validation, feature selection, and preprocessing."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import settings

logger = logging.getLogger(__name__)

CLASS_TARGET = "delay_risk_level"
REGRESSION_TARGET = "risk_probability"
LEAKAGE_COLUMNS = frozenset(
    {
        "project_id",
        CLASS_TARGET,
        REGRESSION_TARGET,
        "estimated_delay_months",
        "actual_delay_days",
        "delay_status",
    }
)
EXPECTED_DATASET_COLUMNS = (
    "project_id",
    "state",
    "project_type",
    "project_start_date",
    "land_area_hectares",
    "total_parcels",
    "private_land_pct",
    "forest_land_pct",
    "govt_land_pct",
    "affected_families_count",
    "rr_settlement_status",
    "forest_clearance_stage",
    "env_clearance_stage",
    "circle_rate_per_sqm",
    "compensation_disbursed_pct",
    "cadastral_survey_discrepancy",
    "active_litigations_count",
    "stay_order_present",
    "inter_agency_nocs_pending",
    "historical_district_delay_score",
    "target_duration_days",
    "actual_delay_days",
    "delay_status",
    REGRESSION_TARGET,
    CLASS_TARGET,
    "estimated_delay_months",
)


class DatasetValidationError(ValueError):
    """Raised when the source data cannot support the requested experiment."""


@dataclass(frozen=True)
class DatasetSplits:
    """Aligned train/test data for both targets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_class_train: pd.Series
    y_class_test: pd.Series
    y_regression_train: pd.Series
    y_regression_test: pd.Series


def _header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return next(csv.reader(handle))
    except (OSError, csv.Error, StopIteration) as exc:
        raise DatasetValidationError(f"Unable to read CSV header from {path}: {exc}") from exc


def load_and_validate_dataset(data_path: Path) -> pd.DataFrame:
    """Load and validate the raw dataset without silently changing its rows."""
    data_path = Path(data_path)
    if not data_path.is_file():
        raise DatasetValidationError(f"Dataset file does not exist: {data_path}")

    columns = _header(data_path)
    duplicates = pd.Index(columns)[pd.Index(columns).duplicated()].unique().tolist()
    if duplicates:
        raise DatasetValidationError(f"Duplicate column names are not allowed: {duplicates}")
    try:
        frame = pd.read_csv(data_path)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DatasetValidationError(f"Unable to load CSV {data_path}: {exc}") from exc

    missing_targets = {CLASS_TARGET, REGRESSION_TARGET} - set(frame.columns)
    if missing_targets:
        raise DatasetValidationError(f"Missing required target columns: {sorted(missing_targets)}")
    if frame.empty:
        raise DatasetValidationError("Dataset contains no records")
    if "project_id" in frame.columns and frame["project_id"].duplicated().any():
        raise DatasetValidationError("project_id must be unique when present")

    excluded = set(LEAKAGE_COLUMNS)
    feature_columns = [column for column in frame.columns if column not in excluded]
    if not feature_columns:
        raise DatasetValidationError("Dataset has no usable feature columns after target exclusion")

    probability = pd.to_numeric(frame[REGRESSION_TARGET], errors="coerce")
    if probability.isna().any():
        raise DatasetValidationError(f"{REGRESSION_TARGET} must contain only numeric values")
    if not probability.between(0.0, 1.0).all():
        raise DatasetValidationError(f"{REGRESSION_TARGET} values must be between 0.0 and 1.0")
    frame[REGRESSION_TARGET] = probability

    labels = frame[CLASS_TARGET]
    if labels.isna().any():
        raise DatasetValidationError(f"{CLASS_TARGET} contains missing values")
    expected = set(settings.expected_class_labels)
    unexpected = sorted(set(labels.astype(str)) - expected)
    if unexpected:
        raise DatasetValidationError(
            f"Unexpected {CLASS_TARGET} categories: {unexpected}; expected {sorted(expected)}"
        )
    frame[CLASS_TARGET] = labels.astype(str)

    missing_count = int(frame.isna().sum().sum())
    logger.info(
        "Validated dataset: rows=%d columns=%d missing_values=%d target_distribution=%s",
        len(frame),
        len(frame.columns),
        missing_count,
        frame[CLASS_TARGET].value_counts().to_dict(),
    )
    return frame


def get_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return model inputs with all known identifiers, targets, and outcomes removed."""
    missing = {CLASS_TARGET, REGRESSION_TARGET} - set(frame.columns)
    if missing:
        raise DatasetValidationError(f"Cannot select model features; missing targets: {sorted(missing)}")
    columns = [column for column in frame.columns if column not in LEAKAGE_COLUMNS]
    if not columns:
        raise DatasetValidationError("No model features remain after leakage exclusions")
    return frame.loc[:, columns].copy()


def separate_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate canonical model features, classification target, and regression target."""
    return get_model_features(frame), frame[CLASS_TARGET].copy(), frame[REGRESSION_TARGET].copy()


def build_preprocessor(
    X: pd.DataFrame,
    *,
    scaler: str = "robust",
) -> ColumnTransformer:
    """Build the sole raw-input transformation used by training and inference."""
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    if not numeric and not categorical:
        raise ValueError("X must contain numeric or categorical columns")
    if scaler not in {"robust", "standard"}:
        raise ValueError("scaler must be 'robust' or 'standard'")
    scaler_step = RobustScaler() if scaler == "robust" else StandardScaler()
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", scaler_step)]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        transformers.append(("categorical", categorical_pipeline, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def split_dataset(
    frame: pd.DataFrame,
    *,
    test_size: float = settings.test_split_fraction,
    random_state: int = settings.random_seed,
) -> DatasetSplits:
    """Create deterministic, aligned stratified splits for both prediction targets."""
    X, y_class, y_regression = separate_targets(frame)
    counts = y_class.value_counts()
    insufficient = counts[counts < 2]
    if not insufficient.empty:
        raise DatasetValidationError(
            f"Each class needs at least two records for stratification: {insufficient.to_dict()}"
        )
    train_idx, test_idx = train_test_split(
        frame.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class,
    )
    return DatasetSplits(
        X_train=X.loc[train_idx],
        X_test=X.loc[test_idx],
        y_class_train=y_class.loc[train_idx],
        y_class_test=y_class.loc[test_idx],
        y_regression_train=y_regression.loc[train_idx],
        y_regression_test=y_regression.loc[test_idx],
    )


def validate_inference_features(
    X: pd.DataFrame,
    expected_columns: Sequence[str],
) -> pd.DataFrame:
    """Validate and order raw inference features before passing them to a fitted transformer."""
    expected = list(expected_columns)
    missing = [column for column in expected if column not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    return X.loc[:, expected].copy()


def load_raw_dataset() -> pd.DataFrame:
    """Load and validate the configured source dataset."""
    return load_and_validate_dataset(settings.dataset_path)
