# DRISHTI Frontend

React/Vite frontend for the DRISHTI full-stack project:
**AI-Powered Predictive Risk System for Land Acquisition Delays in Infrastructure Development.**

The frontend sends the backend's exact 18-feature prediction contract to
`POST /predict`. It does not calculate or fabricate model results in the
browser. Assessment history is retained in `sessionStorage` for the current
browser session only.

## Backend contract

Configure the API base URL with `VITE_API_BASE_URL`. The default is
`http://127.0.0.1:8000`.

The request contains only these model features:

```text
project_type, state, land_area_hectares, total_parcels,
private_land_pct, forest_land_pct, govt_land_pct,
affected_families_count, rr_settlement_status,
forest_clearance_stage, env_clearance_stage, circle_rate_per_sqm,
compensation_disbursed_pct, cadastral_survey_discrepancy,
active_litigations_count, stay_order_present,
inter_agency_nocs_pending, historical_district_delay_score
```

The response is rendered from the API fields:
`delay_risk_level`, `risk_probability`, `top_risk_factors`, and
`model_version`.

## Run locally

```powershell
cd C:\Users\HP\Desktop\data\raw\drishti-ml-service\frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## Production build

```powershell
npm run build
npm run preview
```

## Deploy on Vercel

1. Push the repository to GitHub/GitLab/Bitbucket.
2. In Vercel, choose **Add New Project** and import the repository.
3. Set the project root/directory to `frontend`.
4. Use the detected framework preset **Vite**.
5. Build command: `npm run build`.
6. Output directory: `dist`.
7. Add `VITE_API_BASE_URL` in Vercel project environment variables, pointing
   to the deployed FastAPI service.
8. Ensure the backend CORS configuration allows the Vercel domain.
9. Deploy.

The application uses client-side state and `sessionStorage`; each browser
session has its own temporary assessment history. The model is currently
trained on synthetic data, so results require domain validation before
operational use.
