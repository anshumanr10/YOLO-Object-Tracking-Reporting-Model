from pathlib import Path
from ultralytics import YOLO  # type: ignore
from . import config_loader as config


# Load config once at import time so functions can just read from it.
config.load_config()

_DEFAULT_MODEL: str = config.defaults["model"]
_DEFAULT_CONF: float = float(config.defaults["conf"])
_DEFAULT_FPS: int = int(config.defaults["fps"])
_DEFAULT_TRACKING: bool = bool(config.defaults["tracking"])

def load_model(model: str = _DEFAULT_MODEL) -> YOLO:
    spec = config.models[model]
    return YOLO(spec["weights"])

def confidence_lvl(confidence: float = _DEFAULT_CONF) -> float:
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")
    return float(confidence)

def fps_limit(fps: int = _DEFAULT_FPS) -> int:
    if not isinstance(fps, int) or not (1 <= fps <= 60):
        raise ValueError("FPS must be between 1 and 60")
    return int(fps)

def tracking_enabled(tracking: bool = _DEFAULT_TRACKING) -> bool:
    if not isinstance(tracking, bool):
        raise ValueError("Tracking must be a boolean")
    return bool(tracking)

if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    print(f"Loaded model: {load_model().model_name}")
    print(f"FPS set to: {fps_limit()}")
    print(f"Tracking Enabled?: {tracking_enabled()}")
    print("Yolo model successfully configured.")