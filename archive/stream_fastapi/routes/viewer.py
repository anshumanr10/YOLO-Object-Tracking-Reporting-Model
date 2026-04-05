from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

# Frontend dir: project root / frontend (avoid circular import from stream_fastapi)
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@router.get("/", response_class=FileResponse)
async def viewer_page():
    return FileResponse(_FRONTEND_DIR / "index.html")
