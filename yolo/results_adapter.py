"""
Convert Ultralytics ``Results`` into a small, renderer-friendly overlay structure.

No validation of model outputs; invalid or unexpected shapes surface from tensor access.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class DetectionItem:
    xyxy: Tuple[float, float, float, float]
    cls_id: int
    class_name: str
    conf: float
    track_id: Optional[int] = None
    keypoints_xy: Optional[List[Tuple[float, float]]] = None


@dataclass(frozen=True)
class OverlayState:
    """Latest drawable snapshot for a single frame's inference output."""

    updated_at: float
    detections: List[DetectionItem] = field(default_factory=list)
    orig_shape: Optional[Tuple[int, int]] = None


def _unwrap_result(result: Any) -> Any:
    if result is None:
        raise ValueError("result is None")
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            raise ValueError("empty results list")
        return result[0]
    return result


def _names_map(names: Any) -> Mapping[int, str]:
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {int(i): str(n) for i, n in enumerate(names)} if hasattr(names, "__iter__") else {}


def result_to_overlay(
    result: Any,
    names: Any,
    *,
    timestamp: Optional[float] = None,
) -> OverlayState:
    """
    Build an :class:`OverlayState` from one Ultralytics ``Results`` object (or the first
    element of a list returned by ``track`` / ``predict``).

    ``names`` is typically ``model.names`` (mapping class id -> label).

    Populates ``keypoints_xy`` per detection when ``result.keypoints`` exists and aligns
    by index with ``result.boxes``. Segmentation masks are not converted here.
    """
    r = _unwrap_result(result)
    ts = time.monotonic() if timestamp is None else float(timestamp)
    name_map = _names_map(names)

    orig_shape = None
    if getattr(r, "orig_shape", None) is not None:
        sh = r.orig_shape
        if hasattr(sh, "__len__") and len(sh) >= 2:
            orig_shape = (int(sh[0]), int(sh[1]))

    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return OverlayState(updated_at=ts, detections=[], orig_shape=orig_shape)

    keypoints_xy_list: Optional[List[Optional[List[Tuple[float, float]]]]] = None
    kpts = getattr(r, "keypoints", None)
    if kpts is not None and len(kpts) > 0:
        try:
            xy = kpts.xy
            if xy is not None:
                n = int(xy.shape[0])
                keypoints_xy_list = []
                for i in range(n):
                    row = xy[i]
                    n_j = int(row.shape[0])
                    pts = []
                    for j in range(n_j):
                        v0, v1 = row[j, 0], row[j, 1]
                        x = float(v0.item()) if hasattr(v0, "item") else float(v0)
                        y = float(v1.item()) if hasattr(v1, "item") else float(v1)
                        pts.append((x, y))
                    keypoints_xy_list.append(pts)
        except Exception:
            keypoints_xy_list = None

    detections: List[DetectionItem] = []
    for idx, b in enumerate(boxes):
        cls_id = int(b.cls.item())
        conf_score = float(b.conf.item())
        xyxy_t = b.xyxy[0]
        x1, y1, x2, y2 = (float(xyxy_t[0].item()), float(xyxy_t[1].item()), float(xyxy_t[2].item()), float(xyxy_t[3].item()))
        class_name = name_map.get(cls_id, str(cls_id))
        tid: Optional[int] = None
        if b.id is not None:
            tid = int(b.id.item())

        kpt: Optional[List[Tuple[float, float]]] = None
        if keypoints_xy_list is not None and idx < len(keypoints_xy_list):
            kpt = keypoints_xy_list[idx]

        detections.append(
            DetectionItem(
                xyxy=(x1, y1, x2, y2),
                cls_id=cls_id,
                class_name=class_name,
                conf=conf_score,
                track_id=tid,
                keypoints_xy=kpt,
            )
        )

    return OverlayState(updated_at=ts, detections=detections, orig_shape=orig_shape)
