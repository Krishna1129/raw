# DRISHTI Frontend

Frontend-only React/Vite demonstration for the DRISHTI SIH project:
**AI-Powered Predictive Risk System for Land Acquisition Delays in Infrastructure Development.**

This app intentionally does not call a backend or the ML service. Assessment
results are deterministic, rule-based demo simulations stored in
`sessionStorage`. The UI labels demo data and demo results explicitly.

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
7. No backend, environment variables, database, authentication service, or API URL is required for this frontend-only demo.
8. Deploy.

The application uses client-side state and `sessionStorage`; each browser session
has its own temporary assessment history.
