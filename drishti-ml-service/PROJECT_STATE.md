# Project state

## Phase 0: Workspace initialization

- Project scaffold created.
- Python 3.14 virtual environment created because the Python 3.10 and 3.13 launcher entries were unavailable on disk.
- The supplied source CSV was copied into `data/raw/` without overwriting an existing target.
- Dependencies installed successfully and resolved versions are recorded in `requirements.txt` via `pip freeze`.
- Validation passed: Python 3.14.8, pip 26.2.1, imports, and CSV loading.
- Dataset shape is 25,000 rows x 22 columns.

## Phase 1: Data ingestion and preprocessing

- The actual CSV has 22 columns. Compared with the requested reference schema, it does not contain `project_start_date`, `target_duration_days`, `actual_delay_days`, or `delay_status`.
- `delay_risk_level` is the three-class target (`High`, `Medium`, `Low`); `risk_probability` is the continuous regression target.
- Model features exclude `project_id`, both targets, `estimated_delay_months`, `actual_delay_days`, and `delay_status`. These are identifiers, targets, downstream estimates, or realized outcomes that are unavailable or leaky for prospective prediction.
- Numeric preprocessing uses median imputation and configurable `RobustScaler` (default) or `StandardScaler`. Categorical preprocessing uses most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`.
- Classification-stratified 80/20 splits use one aligned set of row indices and random seed 42 for both targets. No resampling is used.

## Phase 2: Model training and benchmarking

- Classification candidates are Random Forest (`class_weight="balanced"`) and XGBoost multiclass; regression candidates are Random Forest and XGBoost regressors.
- Candidates are selected using a validation split carved only from the training partition: macro F1 for classification and MAE for regression. The held-out test partition is evaluated once after refitting the selected pipeline on all training rows.
- Each selected artifact is a fitted sklearn pipeline containing its own canonical preprocessor and estimator. Regression outputs are clipped to `[0, 1]` at evaluation/inference as a safety constraint, not probability calibration.
- The complete trusted Joblib bundle is written atomically to `models/pipeline_with_preprocessor.pkl`; machine-readable metrics are written under `reports/`.
- Final full-data run selected Random Forest for classification and XGBoost for regression. Held-out metrics: classification accuracy 0.8118, macro F1 0.7572, ROC-AUC OVR 0.9446; regression MAE 0.0361, RMSE 0.0449, R² 0.8763, Spearman 0.9368.
- Custom serialized estimator adapters live in the importable `src.model_bundle` module so artifacts created through `python -m src.train` can be loaded by a separate process.

## Phase 3: SHAP explainability

- `src/explainability.py` creates one reusable `shap.TreeExplainer` from the fitted selected classifier and never refits or preprocesses independently.
- Raw input is ordered with the bundle feature schema, transformed through the fitted `ColumnTransformer`, and explained only after obtaining transformed feature names.
- SHAP list, 2-D, and supported 3-D output layouts are normalized with strict dimension checks. Unsupported shapes or non-finite values raise `ExplainabilityError` and are logged rather than producing incorrect attributions.
- Explanations return at most three signed transformed-feature contributions. Direction values are `increases_predicted_class` or `decreases_predicted_class`; these describe model attribution toward the selected class, not causal effects, percentages, or proof of risk.
- Production SHAP initialization and explanation were exercised against `models/pipeline_with_preprocessor.pkl`; the complete suite passes with 11 tests.

## Phase 4: FastAPI microservice

- The FastAPI lifespan loads and validates the trusted model bundle and reusable SHAP explainer once at startup, then releases references at shutdown.
- `/predict` accepts only the fitted bundle's 18 raw feature columns, rejects extra target/leakage fields, and returns the classifier label, separately generated clipped regression score, model version, and up to three SHAP factors.
- `/health` is a liveness endpoint; `/ready` requires the bundle and explainer to be initialized.
- Validation and internal failures use request IDs and consistent JSON error responses without exposing stack traces. CORS origins are configurable through `DRISHTI_CORS_ORIGINS` and default to localhost React development ports.

## Phase 5: Testing and end-to-end validation

- Added fixture-focused tests for invalid/missing CSVs, missing targets, duplicate identifiers, leakage exclusion, deterministic splits, missing values, unseen categories, training metrics, bundle reload, SHAP output safety, API validation, startup failure, lifecycle loading, and controlled prediction/explanation errors.
- Full pytest validation passes with 22 tests. The complete 25,000-row training job is not required for the test suite; tests use the supplied data to construct small balanced fixtures and reuse the trusted production artifact for API/SHAP integration checks.

## Phase 6: Inference latency and performance

- `scripts/benchmark_inference.py` measures warm model-only classification, model-only regression, SHAP, complete prediction processing, and FastAPI TestClient round-trip latency.
- Each measurement reports request count, mean, median, p95, p99, maximum, and whether SHAP is included. The benchmark distinguishes in-process HTTP overhead from an external network hop and treats 120 ms as a measured objective rather than a guarantee.
- Final benchmark on Windows 11, Python 3.14.8, AMD64/Intel64, using 100 measured requests after 20 warm-ups: classification-only p95 61.5 ms; regression-only p95 8.2 ms; SHAP-only p95 112.8 ms; complete processing with SHAP p95 319.5 ms; in-process HTTP round trip with SHAP p95 197.1 ms. The explanation-enabled service therefore does not meet the 120 ms p95 objective on this machine.
- Model loading is outside request timing and occurs once per TestClient lifespan. The current API performs the required explanations synchronously; the recommended performance architecture is an explicit optional explanation flag or separate explanation endpoint, preserving the current required explanation behavior until that contract is intentionally changed. The result is hardware- and runtime-dependent, not a production SLA.

## Phase 8: Final engineering review

- Code implementation is complete and the full automated suite passes with 22 tests.
- Project-root imports, artifact reload, API startup/readiness, prediction, feature metadata, leakage exclusions, README workflow, and benchmark report structure were rechecked.
- Classification (`delay_risk_level`) and regression (`risk_probability`) remain independent targets with separate fitted pipelines and no target/outcome leakage in either feature set.
- The measured explanation-enabled p95 latency exceeds 120 ms on the recorded machine; this is a performance result, not a code-completeness failure or production SLA.
- The project is not declared production-ready or real-world predictive-valid: the models were trained on synthetic data, SHAP factors are model attributions rather than causal proof, and domain/data validation is still required.
