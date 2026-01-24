import cv2
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime

model = YOLO("yolov8n.pt")
TARGET_CLASS_IDS = {0, 2, 16}  # person, car, dog

# Stats
frame_detection_counts = defaultdict(int)      # counts every detection each frame
unique_ids_by_class = defaultdict(set)         # counts unique tracked objects
detected_classes = set()

def main(cam_index=0, conf=0.35):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {cam_index}")

    print("Running YOLO tracking on webcam. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # TRACK instead of PREDICT (gives b.id)
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

                # Draw
                label = f"{class_name} {conf_score:.2f}"
                if b.id is not None:
                    label += f" ID:{int(b.id.item())}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    write_summary()

def write_summary():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_file = "detections_summary.txt"

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

    print(f"\nSummary written to {output_file}")

if __name__ == "__main__":
    main()
