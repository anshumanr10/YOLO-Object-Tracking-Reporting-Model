from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
from ultralytics import YOLO  # type: ignore

from . import config_loader as config
from . import video_source
from . import video_output
from . import report


# Load config once and cache defaults from it.
config.load_config()
_DEFAULT_CLASS_NAMES: Optional[List[str]] = config.defaults.get("classes")
_DEFAULT_MODEL_KEY: str = config.defaults["model"]
_DEFAULT_CONF: float = float(config.defaults.get("conf", 0.5))
_DEFAULT_PERSIST: bool = bool(config.defaults.get("tracking", True))
_DEFAULT_TRACKER: str = "bytetrack.yaml"
_DEFAULT_VIDEO_SOURCE_SPEC: Dict[str, Any] = config.defaults["source"]


def get_target_class_ids(
    class_names: Optional[List[str]] = _DEFAULT_CLASS_NAMES,
) -> Optional[List[int]]:
    """Resolve default or provided class names to list of class IDs. None = all classes."""
    if not class_names or not config.classifications:
        return None
    ids = [config.classifications[c] for c in class_names if c in config.classifications]
    return ids if ids else None


def tracking_frames(
    video_source_spec: Dict[str, Any] = _DEFAULT_VIDEO_SOURCE_SPEC,
    model_key: str = _DEFAULT_MODEL_KEY,
    conf: float = _DEFAULT_CONF,
    persist: bool = _DEFAULT_PERSIST,
    tracker: str = _DEFAULT_TRACKER,
    classes: Optional[List[int]] = get_target_class_ids(),
    max_frames: Optional[int] = None,
    draw: bool = True,
) -> Generator[Tuple[Any, Any, Any], None, None]:
    """
    Single loop: read frame -> track -> optionally draw -> yield (frame, results, model).
    Caller can aggregate stats (run_tracking) using model.names, or encode and stream (e.g. MJPEG).
    """
    spec = config.models[model_key]
    model = YOLO(spec["weights"])
    conf_val = float(conf)
    target_ids = classes

    cap = video_source.load_video_source(video_source_spec)
    if not cap.isOpened():
        raise RuntimeError("Video source failed to open")

    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_count >= max_frames:
                break

            results = model.track(
                frame,
                conf=conf_val,
                iou=0.7,
                persist=persist,
                tracker=tracker,
                classes=target_ids,
                verbose=False,
            )
            r = results[0]
            if draw:
                video_output.draw_detections(frame, r, model)
            yield (frame, results, model)
            frame_count += 1
    finally:
        cap.release()


def run_tracking(
    video_source_spec: Dict[str, Any] = _DEFAULT_VIDEO_SOURCE_SPEC,
    model_key: str = _DEFAULT_MODEL_KEY,
    conf: float = _DEFAULT_CONF,
    persist: bool = _DEFAULT_PERSIST,
    tracker: str = _DEFAULT_TRACKER,
    classes: Optional[List[int]] = get_target_class_ids(),
    write_summary_file: bool = False,
    summary_path: str = "detections_summary.txt",
    show_display: bool = False,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run YOLO tracking on a video source. Returns a stats dict suitable for API or report.
    Uses tracking_frames() as the single loop; aggregates stats and optionally shows a window.
    """
    frame_detection_counts: Dict[str, int] = defaultdict(int)
    unique_ids_by_class: Dict[str, Set[int]] = defaultdict(set)
    detected_classes: Set[str] = set()
    frame_count = 0

    for frame, results, model in tracking_frames(
        video_source_spec=video_source_spec,
        model_key=model_key,
        conf=conf,
        persist=persist,
        tracker=tracker,
        classes=classes,
        max_frames=max_frames,
        draw=show_display,
    ):
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                class_name = model.names.get(cls_id, str(cls_id))
                detected_classes.add(class_name)
                frame_detection_counts[class_name] += 1
                if b.id is not None:
                    track_id = int(b.id.item())
                    unique_ids_by_class[class_name].add(track_id)
        if show_display:
            cv2.imshow("YOLO Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_count += 1

    if show_display:
        cv2.destroyAllWindows()

    stats = report.summary_dict(
        frame_detection_counts,
        unique_ids_by_class,
        detected_classes,
    )
    stats["frame_count"] = frame_count

    if write_summary_file:
        report.write_summary(
            frame_detection_counts,
            unique_ids_by_class,
            detected_classes,
            output_file=summary_path,
        )

    return stats


if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    stats = run_tracking(
        show_display=True,
        write_summary_file=True,
    )
    print("Tracking finished.")
    print("Summary:", stats)