"""
Backend entry point. Minimal layout: only files that are needed.
Repo root is on sys.path so the yolo package can be imported.
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Repo root (parent of backend/) so "yolo" resolves when running from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.router import api_router
from yolo.video_source import load_video_source


def create_app() -> FastAPI:
    app = FastAPI(
        title="YOLO Tracking API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/")
    def root():
        return {
            "message": "YOLO Tracking API",
            "version": "0.1.0",
            "health": "/api/v1/health",
            "stream": "/api/v1/stream",
            "tracking": "/api/v1/tracking",
            "docs": "/docs",
        }

    @app.get("/stream", include_in_schema=False)
    def stream_redirect():
        return RedirectResponse(url="/api/v1/stream", status_code=302)

    @app.get("/tracking", include_in_schema=False)
    def tracking_redirect():
        return RedirectResponse(url="/api/v1/tracking", status_code=302)

    @app.on_event("startup")
    def startup():
        # Lazy: do not open camera here so /tracking can open it when used alone (single camera).
        app.state.video_source = None

    @app.on_event("shutdown")
    def shutdown():
        if getattr(app.state, "video_source", None) is not None:
            app.state.video_source.release()
            app.state.video_source = None

    return app


app = create_app()