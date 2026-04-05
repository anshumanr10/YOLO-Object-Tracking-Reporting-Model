from datetime import datetime
from typing import Dict, Set, Any


def write_summary(
    frame_detection_counts: Dict[str, int],
    unique_ids_by_class: Dict[str, set],
    detected_classes: Set[str],
    output_file: str = "detections_summary.txt",
) -> str:
    """Write a detection summary to a text file. Returns the output file path."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, "w") as f:
        f.write("YOLO Detection Summary\n")
        f.write("======================\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Frame-level detections (counts per frame):\n")
        if frame_detection_counts:
            for cls in sorted(frame_detection_counts.keys()):
                f.write(f"{cls}: {frame_detection_counts[cls]}\n")
        else:
            f.write("None\n")

        f.write("\nUnique tracked objects (by track ID):\n")
        if unique_ids_by_class:
            for cls in sorted(unique_ids_by_class.keys()):
                f.write(f"{cls}: {len(unique_ids_by_class[cls])}\n")
        else:
            f.write("None\n")

        f.write("\nDetected classes:\n")
        if detected_classes:
            for cls in sorted(detected_classes):
                f.write(f"- {cls}\n")
        else:
            f.write("- None\n")

    return output_file


def summary_dict(
    frame_detection_counts: Dict[str, int],
    unique_ids_by_class: Dict[str, set],
    detected_classes: Set[str],
) -> Dict[str, Any]:
    """Return the same summary as a dict (e.g. for JSON API responses)."""
    return {
        "timestamp": datetime.now().isoformat(),
        "frame_detection_counts": dict(frame_detection_counts),
        "unique_tracked_by_class": {k: len(v) for k, v in unique_ids_by_class.items()},
        "detected_classes": sorted(detected_classes),
    }
