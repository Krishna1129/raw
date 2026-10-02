# Drishti ML Service

FastAPI inference service for the DRISHTI land-acquisition risk models.

The repository contains a reusable preprocessing pipeline, Random Forest and
XGBoost candidate training workflow, SHAP explanations, an inference API, and
automated tests. The current models were trained on the supplied synthetic
dataset, so predictions and explanations require domain validation before
operational use.

## Windows PowerShell quick start

Run all commands from the project root:

```powershell
Set-Location C:\Users\HP\Desktop\data\raw\drishti-ml-service
```

### 1. Create the virtual environment

Python 3.14 was used for the validated environment. If `py -3.14` is not
available on another machine, use an installed compatible Python 3.10+ version
and adjust the command.

```powershell
py -3.14 -m venv .venv
```

### 2. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for the current user, use:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The committed `requirements.txt` contains the resolved package versions used
for the validated environment.

### 4. Place the dataset

Place the source CSV at exactly:

```text
C:\Users\HP\Desktop\data\raw\drishti-ml-service\data\raw\drishti_land_acquisition_dataset.csv
```

The supplied dataset is already present in this workspace. Do not overwrite a
valid dataset unintentionally. The loader validates required targets,
duplicate identifiers, numeric risk scores, class labels, and usable features.

The API accepts a JSON object containing model features; it does **not** accept
a CSV upload for prediction.

### 5. Train the models

```powershell
python -m src.train
```

This command validates the CSV, creates the deterministic stratified split,
selects classification and regression candidates using training-only
validation, evaluates on the untouched test partition, writes reports, and
atomically saves the inference-ready bundle.

Generated outputs:

```text
models\pipeline_with_preprocessor.pkl
reports\classification_metrics.json
reports\regression_metrics.json
```

### 6. Verify the artifact

```powershell
python -c "from pathlib import Path; from src.model_bundle import load_model_bundle; p=Path('models/pipeline_with_preprocessor.pkl'); b=load_model_bundle(p); print(p.resolve()); print('artifact_version:', b['artifact_version']); print('features:', b['feature_columns']); print('selected classification:', b['selected_classification_model']); print('selected regression:', b['selected_regression_model'])"
```

The artifact is a trusted internal Joblib file. Do not load untrusted `.pkl` or
`.joblib` files because deserialization can execute code.

### 7. Run tests

```powershell
python -m pytest -q
```

The tests use small temporary training fixtures where possible and validate
the complete CSV-to-model-to-HTTP path without requiring a full training job
for every test.

### 8. Start FastAPI

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The bundle and reusable SHAP explainer load once during application startup.
Keep this terminal open while using the API. Stop the server with `Ctrl+C`.

### 9. Open the API documentation

Open this URL in a browser:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI schema documents `/predict`, `/health`, `/ready`, request fields,
response fields, validation behavior, risk categories, units, model version,
the separate continuous regression score, and the limitations of
synthetic-data training.

### 10. Check service health and readiness

Liveness:

```text
http://127.0.0.1:8000/health
```

Readiness:

```text
http://127.0.0.1:8000/ready
```

PowerShell checks:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
```

`/ready` succeeds only after the trusted model bundle and SHAP explainer have
loaded successfully.

### 11. Send a test prediction request

The fitted production bundle currently expects these 18 raw feature fields:

```text
project_type
state
land_area_hectares
total_parcels
private_land_pct
forest_land_pct
govt_land_pct
affected_families_count
rr_settlement_status
forest_clearance_stage
env_clearance_stage
circle_rate_per_sqm
compensation_disbursed_pct
cadastral_survey_discrepancy
active_litigations_count
stay_order_present
inter_agency_nocs_pending
historical_district_delay_score
```

Create a request file with values consistent with the Pydantic schema:

```powershell
@'
{
  "project_type": "Industrial Corridor",
  "state": "Tamil Nadu",
  "land_area_hectares": 73.73,
  "total_parcels": 204,
  "private_land_pct": 34.32,
  "forest_land_pct": 23.28,
  "govt_land_pct": 42.39,
  "affected_families_count": 30,
  "rr_settlement_status": "Mostly Completed",
  "forest_clearance_stage": "Stage-1 Approved",
  "env_clearance_stage": "EC Granted",
  "circle_rate_per_sqm": 10428.31,
  "compensation_disbursed_pct": 46.49,
  "cadastral_survey_discrepancy": 0,
  "active_litigations_count": 2,
  "stay_order_present": 0,
  "inter_agency_nocs_pending": 0,
  "historical_district_delay_score": 0.4664
}
'@ | Set-Content -Encoding utf8 .\sample_request.json
```

Send it to `/predict`:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -InFile .\sample_request.json
```

The response contains:

- `delay_risk_level`: `Low`, `Medium`, or `High` from the classifier.
- `risk_probability`: a separate continuous regression score constrained to
  `[0.0, 1.0]`; it is **not** the classifier's probability of the `High` class.
- `top_risk_factors`: up to three SHAP model-attributed contributions.
- `model_version`: the serialized artifact version.

The SHAP factors are model-specific transformed-feature attributions, not
percentages or proof of causation.

### 12. Run the latency benchmark

```powershell
python scripts\benchmark_inference.py --requests 100 --warmup 20
```

The benchmark separately reports classification, regression, SHAP, complete
processing, and in-process HTTP round-trip latency, including mean, median,
p95, p99, maximum, request count, and environment details.

The HTTP benchmark uses FastAPI's in-process `TestClient`; it does not include
the network hop to a separately hosted process. A latency target is a measured
objective, not a production guarantee.

## Configuration

Configuration is centralized in [`src/config.py`](./src/config.py). Copy
[`.env.example`](./.env.example) to `.env` for local overrides. React origins
can be configured with `DRISHTI_CORS_ORIGINS`; the local defaults are
`http://localhost:3000` and `http://localhost:5173`.

## Troubleshooting

### `py` cannot find the requested Python version

Inspect installed launcher entries:

```powershell
py --list
py -0p
```

Use an installed Python 3.10+ version to create `.venv`, then reinstall
dependencies inside that environment.

### PowerShell cannot activate `.venv`

Run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Alternatively, run the environment interpreter directly without activation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Dataset or schema errors during training

Confirm the file exists and inspect its header:

```powershell
Test-Path .\data\raw\drishti_land_acquisition_dataset.csv
Get-Content .\data\raw\drishti_land_acquisition_dataset.csv -TotalCount 2
```

The loader requires `delay_risk_level` and `risk_probability`, rejects
duplicate `project_id` values, validates `risk_probability` in `[0, 1]`, and
expects the three labels `Low`, `Medium`, and `High`.

### API startup or `/ready` failure

Verify the artifact:

```powershell
Test-Path .\models\pipeline_with_preprocessor.pkl
python -c "from src.model_bundle import load_model_bundle; print(load_model_bundle('models/pipeline_with_preprocessor.pkl')['artifact_version'])"
```

Start Uvicorn in the foreground so startup diagnostics are visible:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --log-level info
```

The service logs model-loading failures and prediction diagnostics server-side.
It intentionally returns request IDs and safe messages rather than Python
stack traces or local filesystem paths.

### `/predict` returns HTTP 422

The request must be a JSON object with exactly the 18 fitted feature names.
Do not include target, identifier, or outcome fields such as
`risk_probability`, `delay_risk_level`, `project_id`, or
`estimated_delay_months`. Percentages must be between 0 and 100, counts must
be non-negative, and `stay_order_present` must be 0 or 1.

### Benchmark latency is above 120 ms

Inspect the separate model-only and SHAP measurements rather than hiding the
explanation path. SHAP is measured independently because it can dominate
complete request latency. The documented next architecture is an explicit
optional explanation flag or separate explanation endpoint; required
explanations should not be silently disabled.

### Inspecting logs

For foreground Uvicorn runs, logs appear directly in the server terminal.
The configured project log directory is:

```text
logs\
```

The current service logs to the foreground Uvicorn process; the `logs\`
directory is reserved for future/file-based logging configuration. Request IDs
from error responses can be matched with server-side log entries.
