from pathlib import Path
from . import config_loader as config
from . import model as model_setup
from ultralytics import YOLO # type: ignore
from collections import defaultdict
from datetime import datetime

config.load_config()
model = model_setup.load_model()

model.track(
    source,
    conf=0.25,
    iou=0.7,
    persist=False,
    tracker="bytetrack.yaml",
    classes=None,
    stream=False,
    device=None,
    verbose=False,
)

# Stats
frame_detection_counts = defaultdict(int)      # counts every detection each frame
unique_ids_by_class = defaultdict(set)         # counts unique tracked objects
detected_classes = set()
results = model.track(frame, conf=conf, persist=True, verbose=False)
r = results[0]
if r.boxes is not None and len(r.boxes) > 0:
    for b in r.boxes:
        cls_id = int(b.cls.item())
        if cls_id not in TARGET_CLASS_IDS:
            continue

        conf_score = float(b.conf.item())
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        class_name = model.names.get(cls_id, str(cls_id))

        detected_classes.add(class_name)
        frame_detection_counts[class_name] += 1

        # Unique object counting via track id
        if b.id is not None:
            track_id = int(b.id.item())
            unique_ids_by_class[class_name].add(track_id)