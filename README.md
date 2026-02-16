# J.A.R.V.I.S. - Desktop Automation Platform

J.A.R.V.I.S. (Just A Rather Very Intelligent System) is a comprehensive desktop automation platform with neural core intelligence.

## Features

- 🧠 Neural Core with Gemini AI integration
- 🖥️ Desktop automation (window management, app launching)
- 📊 System monitoring and analytics
- ⚡ Productivity workflows
- 🔒 Security monitoring (Iron Shield)
- 🧹 File organization (Ghost Sweep)
- 🏗️ 3D processing pipeline (Forge)

## Quick Start

### Full Local Setup (Recommended)

1. **Install Dependencies**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt

   # Install Node.js dependencies
   npm install
   ```

2. **Start the Platform**
   ```bash
   # Option 1: Use the workflow
   # Run /start-app workflow in your IDE

   # Option 2: Manual start
   # Terminal 1: Start the backend
   python3 -m uvicorn jason.core.api_server:app --host 127.0.0.1 --port 8000 --reload

   # Terminal 2: Start the frontend
   npm run dev
   ```

3. **Open in Browser**
   - Frontend: http://localhost:5173/
   - Backend API: http://127.0.0.1:8000/status

### Demo Mode (GitHub Pages)

The GitHub Pages deployment runs in demo mode since it can't host the Python backend. All automation commands will simulate responses.

**Note**: For full functionality (actual desktop automation), you must run the app locally as described above.

## Architecture

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python
- **AI Engine**: Google Gemini 2.0 Flash
- **Workflow Engine**: LangGraph + CrewAI

## Workflows

Use these built-in workflows for common tasks:

- `/start-app` - Start the full platform (backend + frontend)
- `/build-app` - Build for production

## Configuration

### API Keys (Optional)

For full AI capabilities, add these to your environment or `config.yaml`:

- `GEMINI_API_KEY`: Google Gemini API key for neural core
- Other API keys for travel booking, research, etc.

### Platform Support

- macOS (primary)
- Windows/Linux (experimental)

## Development

```bash
# Backend development
python3 -m uvicorn jason.core.api_server:app --reload --host 0.0.0.0 --port 8000

# Frontend development
npm run dev
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.