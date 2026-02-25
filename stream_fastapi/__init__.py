from fastapi import FastAPI

from .routes import viewer, websocket


def create_app() -> FastAPI:
    app = FastAPI(title="Stream server")
    app.include_router(viewer.router)
    app.include_router(websocket.router)
    return app


app = create_app()
