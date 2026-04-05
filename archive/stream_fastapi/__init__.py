from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes import camera_options, viewer, websocket, tracking_viewer

# Frontend files (HTML, CSS, JS) live in project root / frontend
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(title="Stream server")
    # API / routes
    app.include_router(camera_options.router)
    app.include_router(viewer.router)
    app.include_router(websocket.router)
    app.include_router(tracking_viewer.router)
    # Static files for frontend
    if _FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
    return app


app = create_app()
