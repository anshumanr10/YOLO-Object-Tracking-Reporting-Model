#!/usr/bin/env python3
"""
Conventional FastAPI entrypoint for the WebRTC stream server.

The actual application, routes, and configuration live in the
`stream_fastapi` package.
"""

from stream_fastapi import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

