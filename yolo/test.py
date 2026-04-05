"""
Smoke-test the ``yolo`` package: webcam → ``YOLOTracker.track_frame`` → ``ResultsSession`` → draw + console report.

Run from repository root::

    python -m yolo.test

Uses only this package plus OpenCV and Ultralytics. Requires a webcam for the default path.
Default model weights download on first run.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Generator, Iterable, Optional

from . import ResultsSession, YOLOTracker
from .results_adapter import OverlayState


def webcam_frame_stream(camera_index: int) -> Generator[Any, None, None]:
    """BGR frames from a local OpenCV camera index (minimal iterator for this test module)."""
    import cv2

    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame
    finally:
        cap.release()


def draw_overlay_bgr(frame: Any, overlay: Optional[OverlayState], *, window_name: str) -> None:
    """Draw ``OverlayState`` on a copy of ``frame`` and show in a window."""
    import cv2

    if overlay is None or not overlay.detections:
        cv2.imshow(window_name, frame)
        return
    out = frame.copy()
    for d in overlay.detections:
        x1, y1, x2, y2 = (int(d.xyxy[0]), int(d.xyxy[1]), int(d.xyxy[2]), int(d.xyxy[3]))
        label = f"{d.class_name} {d.conf:.2f}"
        if d.track_id is not None:
            label += f" id:{d.track_id}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.imshow(window_name, out)


def run(
    *,
    camera_index: int,
    model: str,
    max_frames: Optional[int],
    stats_interval_s: float,
) -> None:
    import cv2

    tracker = YOLOTracker(model)
    names = tracker.model.names
    session = ResultsSession()

    frame_iter: Iterable[Any] = webcam_frame_stream(camera_index)
    window = "yolo test (q to quit)"
    last_stats = time.monotonic()
    n = 0

    try:
        for frame in frame_iter:
            results = tracker.track_frame(frame)
            session.on_result(results, names)
            draw_overlay_bgr(frame, session.latest_overlay, window_name=window)

            now = time.monotonic()
            if now - last_stats >= stats_interval_s:
                live = session.live
                print(
                    f"[live] n={session.result_count} detections={len(live)} "
                    f"unique_tracks_total={session.total_unique_tracked_objects}",
                    flush=True,
                )
                last_stats = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            n += 1
            if max_frames is not None and n >= max_frames:
                break
    finally:
        cv2.destroyAllWindows()

    print("summary:", session.summary_dict(), flush=True)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Webcam smoke test for the yolo package.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0).")
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics weights path or hub name (default yolov8n.pt).",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (default: until 'q').")
    p.add_argument(
        "--stats-interval",
        type=float,
        default=1.0,
        help="Seconds between live stat lines (default 1.0).",
    )
    args = p.parse_args(argv)

    try:
        run(
            camera_index=args.camera,
            model=args.model,
            max_frames=args.max_frames,
            stats_interval_s=args.stats_interval,
        )
    except KeyboardInterrupt:
        print("interrupted", flush=True)
        return 130
    except Exception as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
