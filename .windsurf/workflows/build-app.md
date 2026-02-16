---
description: Build the J.A.R.V.I.S. Platform for Production
---

This workflow builds the frontend and ensures the backend is ready.

1. Build the Vite frontend
// turbo
run_command: npm run build

2. Check backend health
// turbo
run_command: curl http://127.0.0.1:8000/status
