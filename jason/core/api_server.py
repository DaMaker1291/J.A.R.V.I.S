"""
J.A.S.O.N. API Server - FastAPI for local command execution
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import threading

app = FastAPI(title="J.A.S.O.N. API", description="Local execution API for J.A.S.O.N.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:5174"],
    allow_credentials=True,
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

@app.on_event("startup")
async def startup_event():
    """Initialize managers on startup"""
    # Managers are now lazy-loaded in the endpoints
    pass

@app.post("/execute")
async def execute_command(request: CommandRequest):
    """Execute a command through J.A.S.O.N."""
    try:
        global swarm_manager
        if swarm_manager is None:
            from jason.core.swarm import SwarmManager
            swarm_manager = SwarmManager()

        # Execute task through LangGraph
        result = swarm_manager.execute_task(request.command)

        return {"result": result, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution failed: {str(e)}")

@app.post("/clarify")
async def clarify_command(request: ClarificationRequest):
    """Provide clarification for ambiguous command"""
    try:
        global swarm_manager
        if swarm_manager is None:
            from jason.core.swarm import SwarmManager
            swarm_manager = SwarmManager()

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
