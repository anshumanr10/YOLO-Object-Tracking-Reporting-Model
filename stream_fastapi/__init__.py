from fastapi import FastAPI

from .routes import viewer, websocket, tracking_viewer


def create_app() -> FastAPI:
    app = FastAPI(title="Stream server")
    app.include_router(viewer.router)
    app.include_router(websocket.router)
    app.include_router(tracking_viewer.router)
    return app


app = create_app()
