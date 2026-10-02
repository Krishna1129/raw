"""Centralized project configuration with paths independent of the CWD."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    project_root: Path = PROJECT_ROOT
    dataset_path: Path = PROJECT_ROOT / "data" / "raw" / "drishti_land_acquisition_dataset.csv"
    model_artifact_path: Path = PROJECT_ROOT / "models" / "pipeline_with_preprocessor.pkl"
    reports_directory: Path = PROJECT_ROOT / "reports"
    random_seed: int = 42
    test_split_fraction: float = 0.2
    expected_class_labels: tuple[str, ...] = ("Low", "Medium", "High")
    api_title: str = "Drishti ML Service"
    api_version: str = "0.1.0"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")
    log_level: str = "INFO"
    log_directory: Path = PROJECT_ROOT / "logs"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
