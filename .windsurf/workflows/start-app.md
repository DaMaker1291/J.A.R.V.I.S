---
description: Start the J.A.R.V.I.S. Platform (Backend + Frontend)
---

This workflow will start both the FastAPI backend and the Vite frontend for development.

1. Install Python dependencies
// turbo
run_command: pip install -r requirements.txt

2. Install Node.js dependencies
// turbo
run_command: npm install

3. Start the FastAPI Neural Core (Backend)
// turbo
run_command: python3 -m uvicorn jason.core.api_server:app --host 127.0.0.1 --port 8000 --reload

4. Start the React UI (Frontend)
// turbo
run_command: npm run dev

5. Open the Dashboard
// turbo
browser_navigate: http://localhost:5173/J.A.R.V.I.S/
