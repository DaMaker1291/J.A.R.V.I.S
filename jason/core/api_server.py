"""
J.A.S.O.N. API Server - FastAPI for local command execution
"""

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import threading
import os
import sys
from pathlib import Path

# Add project root to path for absolute imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import anyio

import psutil

import yaml

from fastapi import UploadFile, File

app = FastAPI(title="J.A.S.O.N. API", description="Local execution API for J.A.S.O.N.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    
    api_keys = config.get("api_keys")
    if isinstance(api_keys, dict):
        cfg_key = api_keys.get("gemini")
        if isinstance(cfg_key, str) and cfg_key.strip():
            return cfg_key.strip()
    return ""

@app.on_event("startup")
async def startup_event():
    """Initialize managers on startup"""
    # Managers are now lazy-loaded in the endpoints
    pass

async def get_status():
    """Get API status"""
    return {"status": "ok", "message": "J.A.R.V.I.S. API is running"}

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
            # Set environment variables for libraries that need them
            if gemini_api_key:
                os.environ["GEMINI_API_KEY"] = gemini_api_key
                os.environ["GOOGLE_API_KEY"] = gemini_api_key
            swarm_manager = SwarmManager(gemini_api_key=gemini_api_key, config=config)

        # Execute task through LangGraph
        result = await anyio.to_thread.run_sync(swarm_manager.process_command, request.command)
        return {"result": result}

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"ERROR in execute_command: {error_detail}")
        with open("/tmp/error_traceback.log", "w") as f:
            f.write(error_detail)
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

@app.get("/")
async def root():
    """Root endpoint returning API status"""
    return await get_status()

@app.get("/status")
async def status():
    """Status endpoint for frontend health check"""
    return await get_status()

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        processes = len(psutil.pids())
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": memory.used,
            "memory_total": memory.total,
            "disk_percent": disk.percent,
            "processes": processes
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/search-products")
async def search_products(q: str = "", limit: int = 10):
    """Search AliExpress products for dropshipping."""
    try:
        from jason.tools.browser_agent import BrowserAgent
        import urllib.parse

        async with BrowserAgent() as agent:
            search_url = f"https://www.aliexpress.com/wholesale?SearchText={urllib.parse.quote(q)}"
            await agent.navigate(search_url)
            
            # Wait for products to load
            await agent.wait_for_element('.product-item, [data-product-id]', 15000)
            
            # Extract products
            products = await agent.page.evaluate(f'''
                const products = [];
                const limit = {limit};
                document.querySelectorAll('.product-item, [data-product-id], .list-item').forEach((item, index) => {{
                    if (products.length >= limit) return;
                    const title = item.querySelector('.product-title, .title, h3')?.textContent?.trim();
                    const price = item.querySelector('.price, .product-price')?.textContent?.trim();
                    const image = item.querySelector('img')?.src;
                    const url = item.querySelector('a')?.href;
                    if (title && price) {{
                        products.push({{
                            id: index + 1,
                            title,
                            price,
                            image,
                            url
                        }});
                    }}
                }});
                return products;
            ''')
            
            return {"products": products}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/analyze-handwriting")
async def analyze_handwriting(image: UploadFile = File(...)):
    """Analyze handwriting style from worksheet image using config defaults."""
    try:
        config = _load_config()
        handwriting_config = config.get("handwriting", {})
        
        # Use real config values for style
        style = {
            "slant": handwriting_config.get("default_slant", -5),
            "pressure": handwriting_config.get("default_pressure", 0.8),
            "noise": handwriting_config.get("default_noise", 0.1),
            "fields": ["Name", "Date", "Answer1", "Answer2"]  # Default fields for worksheets
        }
        
        return {"style": style}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api_server()
