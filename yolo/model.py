from pathlib import Path
from . import config_loader as config
from ultralytics import YOLO # type: ignore

def load_model(model: str | None = None) -> YOLO:
    if not model:
        config.load_config()
    
    key = model or config.defaults["model"]
    spec = config.models[key]
    return YOLO(spec["weights"])

def confidence_lvl(confidence: float | None = None) -> float:
    if confidence is None:
        config.load_config()
        confidence = config.defaults["conf"]
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0.0 and 1.0")
    return float(confidence)

def fps_limit(fps: int | None = None) -> int:
    if fps is None:
        config.load_config()
        fps = config.defaults["fps"]
    if not isinstance(fps, int) or not (1 <= fps <= 60):
        raise ValueError("FPS must be between 1 and 60")
    return int(fps)

def tracking_enabled(tracking: bool | None = None) -> bool:
    if tracking is None:
        config.load_config()
        tracking = config.defaults["tracking"]
    if not isinstance(tracking, bool):
        raise ValueError("Tracking must be a boolean")
    return bool(tracking)

if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    print(f"Loaded model: {load_model().model_name}")
    print(f"FPS set to: {fps_limit()}")
    print(f"Tracking Enabled?: {tracking_enabled()}")
    print("Yolo model successfully configured.")