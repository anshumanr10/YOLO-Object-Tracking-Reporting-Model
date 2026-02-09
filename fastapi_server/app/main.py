# pyright: reportMissingImports=false

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi_server.app.api.v1.router import api_router

# Repo root: so "yolo" package and "config/" are found when app runs from any cwd
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

app = FastAPI(
    title="YOLO Tracking API",
    version="0.1.0",
)

# So a frontend on another origin (e.g. localhost:3000) can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to specific origins in production, e.g. ["https://your-app.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """Use repo root as cwd so yolo config_loader finds config/."""
    os.chdir(REPO_ROOT)


@app.get("/")
def root():
    return {
        "message": "YOLO Tracking API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "testbench": "/testbench",
    }


@app.get("/testbench", include_in_schema=False)
def testbench_redirect():
    """Redirect /testbench to /testbench/ so static index.html is served."""
    return RedirectResponse(url="/testbench/", status_code=302)


app.include_router(api_router, prefix="/api/v1")
app.mount("/testbench", StaticFiles(directory=str(REPO_ROOT / "frontend_testbench"), html=True), name="testbench")
