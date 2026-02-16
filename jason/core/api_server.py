"""
J.A.S.O.N. API Server - FastAPI for local command execution
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import threading
import os
from pathlib import Path

import yaml

app = FastAPI(title="J.A.S.O.N. API", description="Local execution API for J.A.S.O.N.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://.*\\.github\\.io$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    command: str
    parameters: Optional[Dict[str, Any]] = None
    require_clarification: Optional[bool] = True

class ClarificationRequest(BaseModel):
    selected_option: str
    task_id: str

# Global instances
swarm_manager = None
vision_manager = None
audio_manager = None
overlay = None


def _load_config() -> Dict[str, Any]:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    return {}


def _get_gemini_api_key(config: Dict[str, Any]) -> str:
    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    cfg_key = (config.get("api_keys") or {}).get("gemini")
    if isinstance(cfg_key, str) and cfg_key.strip():
        return cfg_key.strip()
    return ""

@app.on_event("startup")
async def startup_event():
    """Initialize managers on startup"""
    # Managers are now lazy-loaded in the endpoints
    pass

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to capture all errors"""
    import traceback
    error_detail = traceback.format_exc()
    print(f"CRITICAL ERROR: {error_detail}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}", "traceback": error_detail}
    )

@app.post("/execute")
async def execute_command(request: CommandRequest):
    """Execute a command through J.A.S.O.N."""
    try:
        global swarm_manager
        if swarm_manager is None:
            from jason.core.swarm import SwarmManager
            config = _load_config()
            gemini_api_key = _get_gemini_api_key(config)
            swarm_manager = SwarmManager(gemini_api_key=gemini_api_key, config=config)

        # Execute task through LangGraph
        print(f"Executing command: {request.command}")
        result = swarm_manager.process_command(request.command)
        print(f"Execution result: {result}")

        return {"result": result, "status": "success"}

    except Exception as e:
        import traceback
        print(f"Error in execute_command: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.post("/clarify")
async def clarify_command(request: ClarificationRequest):
    """Provide clarification for ambiguous command"""
    try:
        global swarm_manager
        if swarm_manager is None:
            from jason.core.swarm import SwarmManager
            config = _load_config()
            gemini_api_key = _get_gemini_api_key(config)
            swarm_manager = SwarmManager(gemini_api_key=gemini_api_key, config=config)

        # Update task with clarification
        # This would need to be integrated with the LangGraph state management
        result = f"Clarified: {request.selected_option} for task {request.task_id}"

        return {"result": result, "status": "clarified"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clarification failed: {str(e)}")

@app.post("/overlay/highlight")
async def highlight_areas(areas: Dict[str, Any]):
    """Highlight areas on screen for selection"""
    try:
        global overlay
        if overlay is None:
            from jason.core.overlay import ScreenOverlay
            overlay = ScreenOverlay()

        # areas should contain list of rectangles: [{"x": x, "y": y, "w": w, "h": h, "label": label}]
        rectangles = areas.get("rectangles", [])

        # Start overlay in separate thread
        overlay_thread = threading.Thread(target=lambda: overlay.show_overlay(rectangles))
        overlay_thread.daemon = True
        overlay_thread.start()

        return {"status": "overlay_shown", "rectangles": len(rectangles)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overlay failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "version": "OMNI-2.1",
        "modules": {
            "swarm": "lazy_loaded",
            "vision": "lazy_loaded",
            "audio": "lazy_loaded",
            "overlay": "lazy_loaded"
        }
    }

def start_api_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server"""
    uvicorn.run(app, host=host, port=port)
