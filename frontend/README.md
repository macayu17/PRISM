# NeuroAssess Frontend

Vite/React client for the Parkinson's Disease Assessment Portal. It renders the
clinical dashboard, assessment workflow, medical document manager, and digital
twin workspace.

## Local Development

Start the Flask API from the project root first:

```bash
python start_server.py
```

Then start the Vite client:

```bash
npm install
npm run dev
```

The Vite dev server proxies `/api/*` requests to `http://localhost:5000`.

## Verification

```bash
npm run lint
npm run build
```
