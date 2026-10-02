"""Pydantic v2 request and response contracts for the inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RiskLevel = Literal["Low", "Medium", "High"]
Direction = Literal["increases_predicted_class", "decreases_predicted_class"]


class PredictionRequest(BaseModel):
    """Exactly the 18 raw features used by the fitted production pipelines."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_type: str | None = None
    state: str | None = None
    land_area_hectares: float | None = Field(default=None, ge=0)
    total_parcels: int | None = Field(default=None, ge=0)
    private_land_pct: float | None = Field(default=None, ge=0, le=100)
    forest_land_pct: float | None = Field(default=None, ge=0, le=100)
    govt_land_pct: float | None = Field(default=None, ge=0, le=100)
    affected_families_count: int | None = Field(default=None, ge=0)
    rr_settlement_status: str | None = None
    forest_clearance_stage: str | None = None
    env_clearance_stage: str | None = None
    circle_rate_per_sqm: float | None = Field(default=None, ge=0)
    compensation_disbursed_pct: float | None = Field(default=None, ge=0, le=100)
    cadastral_survey_discrepancy: int | None = Field(default=None, ge=0)
    active_litigations_count: int | None = Field(default=None, ge=0)
    stay_order_present: int | None = Field(default=None, ge=0, le=1)
    inter_agency_nocs_pending: int | None = Field(default=None, ge=0)
    historical_district_delay_score: float | None = Field(default=None, ge=0)


class RiskFactor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feature: str
    contribution: float
    direction: Direction
    display_name: str
    description: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delay_risk_level: RiskLevel
    risk_probability: float = Field(ge=0, le=1)
    top_risk_factors: list[RiskFactor] = Field(default_factory=list, max_length=3)
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready"]
    model_version: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str
    details: list[dict[str, object]] | None = None
