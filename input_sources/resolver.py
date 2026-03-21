from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import cv2

from .source_opencv import cv_frame_stream
from .source_picamera import picamera2_frame_stream


@dataclass(frozen=True)
class InputSourceSpec:
    """Normalized device-agnostic source spec used by tracking."""

    source_type: str
    capture_ref: Union[int, str]
    sensor_mode_index: Optional[int] = None


def source_spec_from_flat(
    flat: Mapping[str, Any],
    *,
    default_device_index: int = 0,
) -> InputSourceSpec:
    """
    Build InputSourceSpec from a flat options dict (API / merged session state).

    Expected keys: source_type, camera_index, optional sensor_mode_index, url, values.
    """
    cfg: dict[str, Any] = {"type": flat.get("source_type") or "PiCamera"}
    if "camera_index" in flat and flat["camera_index"] is not None:
        cfg["camera_index"] = flat["camera_index"]
    if "sensor_mode_index" in flat:
        cfg["sensor_mode_index"] = flat["sensor_mode_index"]
    if flat.get("url"):
        cfg["url"] = flat["url"]
    if flat.get("values") is not None:
        cfg["values"] = flat["values"]
    return resolve_source_spec(cfg, default_device_index=default_device_index)


def resolve_source_spec(source_cfg: Mapping[str, Any], *, default_device_index: int = 0) -> InputSourceSpec:
    """Resolve defaults.yaml source config into a normalized source spec."""
    src_type = str(source_cfg.get("type") or "PiCamera").strip() or "PiCamera"
    sensor_mode_index = source_cfg.get("sensor_mode_index")
    sensor_mode = int(sensor_mode_index) if isinstance(sensor_mode_index, (int, float)) else None

    if src_type == "PiCamera":
        camera_index = source_cfg.get("camera_index")
        cap = int(camera_index) if isinstance(camera_index, (int, float)) else int(default_device_index)
        return InputSourceSpec(source_type="PiCamera", capture_ref=cap, sensor_mode_index=sensor_mode)

    cap = _resolve_opencv_capture_ref(source_cfg, default_device_index=default_device_index)
    return InputSourceSpec(source_type=src_type, capture_ref=cap, sensor_mode_index=None)


def patch_source_spec(
    spec: InputSourceSpec,
    *,
    source_type: Optional[str] = None,
    camera_index: Optional[Union[int, str]] = None,
    sensor_mode_index: Optional[int] = None,
) -> InputSourceSpec:
    """Apply partial API updates to an existing source spec."""
    stype = source_type.strip() if isinstance(source_type, str) and source_type.strip() else spec.source_type
    cap = spec.capture_ref if camera_index is None else _normalize_camera_index(camera_index)
    sensor = spec.sensor_mode_index if sensor_mode_index is None else sensor_mode_index

    if stype == "PiCamera":
        cap_idx = int(cap) if isinstance(cap, int) else _parse_int_or_default(str(cap), 0)
        return InputSourceSpec(source_type="PiCamera", capture_ref=cap_idx, sensor_mode_index=sensor)

    return InputSourceSpec(source_type=stype, capture_ref=cap, sensor_mode_index=None)


def make_frame_stream(spec: InputSourceSpec, *, target_fps: Optional[int]) -> Any:
    """Create frame generator yielding BGR frames based on resolved source spec."""
    if spec.source_type == "PiCamera":
        kwargs: dict[str, Any] = {"camera_index": int(spec.capture_ref)}
        if target_fps is not None:
            kwargs["target_fps"] = int(target_fps)
        if spec.sensor_mode_index is not None:
            kwargs["sensor_mode_index"] = int(spec.sensor_mode_index)
        return picamera2_frame_stream(**kwargs)

    capture = cv2.VideoCapture(spec.capture_ref)
    kwargs = {"capture": capture}
    if target_fps is not None:
        kwargs["target_fps"] = int(target_fps)
    return cv_frame_stream(**kwargs)


def api_options_from_spec(spec: InputSourceSpec) -> dict[str, Any]:
    return {
        "source_type": spec.source_type,
        "camera_index": spec.capture_ref,
        "sensor_mode_index": spec.sensor_mode_index,
    }


def _resolve_opencv_capture_ref(source_cfg: Mapping[str, Any], *, default_device_index: int) -> Union[int, str]:
    url = source_cfg.get("url")
    if isinstance(url, str) and url.strip():
        return url.strip()

    camera_index = source_cfg.get("camera_index")
    if camera_index is not None:
        return _normalize_camera_index(camera_index)

    values = source_cfg.get("values") if isinstance(source_cfg.get("values"), dict) else {}
    src_type = str(source_cfg.get("type") or "").strip()

    if src_type == "Webcam" and isinstance(values.get("int"), (int, float)):
        return int(values["int"])
    if src_type in {"RTSP", "RTMP", "HTML"} and isinstance(values.get("url"), str) and values["url"].strip():
        return values["url"].strip()
    if src_type == "File" and isinstance(values.get("path"), str) and values["path"].strip():
        return values["path"].strip()

    return int(default_device_index)


def _normalize_camera_index(camera_index: Any) -> Union[int, str]:
    if isinstance(camera_index, (int, float)):
        return int(camera_index)
    if isinstance(camera_index, str):
        s = camera_index.strip()
        if not s:
            return 0
        try:
            return int(s)
        except ValueError:
            return s
    return 0


def _parse_int_or_default(value: str, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
