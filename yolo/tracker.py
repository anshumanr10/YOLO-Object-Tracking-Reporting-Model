from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
from . import config_loader as config
from . import model as model_setup
from . import video_source
from . import video_output
from . import report
from ultralytics import YOLO  # type: ignore


def get_target_class_ids() -> Optional[List[int]]:
    """Resolve config default 'classes' (list of names) to list of class IDs. None = all classes."""
    config.load_config()
    names = config.defaults.get("classes")
    if not names or not config.classifications:
        return None
    ids = [config.classifications[c] for c in names if c in config.classifications]
    return ids if ids else None


def tracking_frames(
    video_source_spec: Optional[Dict[str, Any]] = None,
    model_key: Optional[str] = None,
    conf: Optional[float] = None,
    persist: bool = True,
    tracker: str = "bytetrack.yaml",
    classes: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
    draw: bool = True,
) -> Generator[Tuple[Any, Any, Any], None, None]:
    """
    Single loop: read frame -> track -> optionally draw -> yield (frame, results, model).
    Caller can aggregate stats (run_tracking) using model.names, or encode and stream (e.g. MJPEG).
    """
    config.load_config()
    model = model_setup.load_model(model_key)
    conf_val = model_setup.confidence_lvl(conf)
    target_ids = classes if classes is not None else get_target_class_ids()

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
    video_source_spec: Optional[Dict[str, Any]] = None,
    model_key: Optional[str] = None,
    conf: Optional[float] = None,
    persist: bool = True,
    tracker: str = "bytetrack.yaml",
    classes: Optional[List[int]] = None,
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