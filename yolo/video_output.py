import cv2
from typing import Any

# Ultralytics Result has .boxes (Boxes object) with .cls, .conf, .xyxy, .id
# model.names is dict id -> class name


def draw_detections(
    frame: Any,
    result: Any,
    model: Any,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> Any:
    """
    Draw bounding boxes and labels on a frame from one YOLO result.
    result is typically results[0] from model.track() or model.predict().
    Returns the same frame (modified in place).
    """
    if result.boxes is None or len(result.boxes) == 0:
        return frame

    for b in result.boxes:
        cls_id = int(b.cls.item())
        conf_score = float(b.conf.item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        class_name = model.names.get(cls_id, str(cls_id))

        label = f"{class_name} {conf_score:.2f}"
        if b.id is not None:
            label += f" ID:{int(b.id.item())}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            thickness,
        )
    return frame
