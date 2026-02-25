#!/usr/bin/env python3
"""Minimal script that opens the Pi camera (libcamera/picamera2), takes a picture, saves it, then stops."""
import sys

import cv2
from picamera2 import Picamera2

OUTPUT_FILE = "open_pi_camera_capture.jpg"

def main():
    print("Opening Pi camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": (1280, 720)})
    picam2.configure(config)
    picam2.start()
    print("Taking picture...")
    frame = picam2.capture_array()
    cv2.imwrite(OUTPUT_FILE, frame)
    print(f"Saved {OUTPUT_FILE}")
    picam2.stop()
    picam2.close()
    print("Done.")

if __name__ == "__main__":
    main()
    sys.exit(0)
