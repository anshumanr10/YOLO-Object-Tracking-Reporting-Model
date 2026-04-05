"""
Mutable session over Ultralytics ``Results``: overlay, live view, accumulated unique track keys.

Accumulated stats count distinct (class_name, track_id) pairs first seen in the session, not per-frame box counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .results_adapter import DetectionItem, OverlayState, result_to_overlay


def _sorted_live_detections(overlay: OverlayState) -> List[DetectionItem]:
    """Stable order for display: class name, then tracked before untracked, then track id."""
    return sorted(
        overlay.detections,
        key=lambda d: (
            d.class_name,
            d.track_id is None,
            d.track_id if d.track_id is not None else 0,
        ),
    )


@dataclass
class ResultsSession:
    """
    - ``latest_overlay``: last :class:`OverlayState` from ``on_result`` (same cadence as box updates).
    - ``live``: sorted copy of ``latest_overlay.detections`` (each row is a :class:`DetectionItem`).
    - Accumulated: set of ``(class_name, track_id)`` for each track id seen for the first time.
      Detections with ``track_id is None`` do not contribute to accumulated stats.
    """

    latest_overlay: Optional[OverlayState] = None
    result_count: int = 0
    _accumulated_keys: Set[Tuple[str, int]] = field(default_factory=set, repr=False)

    @property
    def live(self) -> List[DetectionItem]:
        """Current on-screen detections (sorted); empty if no result processed yet."""
        if self.latest_overlay is None:
            return []
        return _sorted_live_detections(self.latest_overlay)

    @property
    def accumulated_unique_keys(self) -> List[Tuple[str, int]]:
        """Sorted copy of distinct (class_name, track_id) seen so far."""
        return sorted(self._accumulated_keys)

    @property
    def total_unique_tracked_objects(self) -> int:
        return len(self._accumulated_keys)

    def on_result(self, result: Any, names: Any) -> None:
        overlay = result_to_overlay(result, names)
        self.latest_overlay = overlay
        self.result_count += 1
        for d in overlay.detections:
            if d.track_id is not None:
                self._accumulated_keys.add((d.class_name, int(d.track_id)))

    def consume_results(self, results_iter: Iterable[Any], names: Any) -> None:
        for r in results_iter:
            self.on_result(r, names)

    def reset(self) -> None:
        """Clear overlay, counts, and accumulated keys."""
        self.latest_overlay = None
        self.result_count = 0
        self._accumulated_keys.clear()

    def summary_dict(self) -> Dict[str, Any]:
        """JSON-friendly snapshot (includes current ``live`` and accumulated unique ids by class)."""
        by_class: Dict[str, List[int]] = {}
        for class_name, tid in self.accumulated_unique_keys:
            by_class.setdefault(class_name, []).append(tid)
        return {
            "timestamp": datetime.now().isoformat(),
            "result_count": self.result_count,
            "live": [
                {
                    "class_name": d.class_name,
                    "cls_id": d.cls_id,
                    "conf": d.conf,
                    "track_id": d.track_id,
                }
                for d in self.live
            ],
            "accumulated_unique_track_ids_by_class": by_class,
            "total_unique_tracked_objects": self.total_unique_tracked_objects,
        }
