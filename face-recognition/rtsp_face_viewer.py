#!/usr/bin/env python3
"""
RTSP -> YOLOv8 face model -> live window with boxes.
Standalone; does not import project packages.
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "yolov8n-face.pt"


def main() -> None:
    url = input("RTSP stream URL: ").strip()
    if not url:
        print("No URL given.")
        return

    weights = DEFAULT_WEIGHTS
    if not weights.is_file():
        print(f"Model not found: {weights}")
        return

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Could not open stream.")
        return

    window = "Face detections (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream ended or frame read failed; exiting.")
                break

            results = model.predict(frame, verbose=False)
            out = results[0].plot()

            cv2.imshow(window, out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
