#!/usr/bin/env python3
"""
List all available cameras using Picamera2, prompt the user to select one,
then either:
- run the camera with a NULL preview (no display) for 5 seconds, or
- record a 5 second video to an MP4 file (H.264, encoder chosen automatically).
"""

import sys
import time

from picamera2 import Picamera2, Preview

# Pi hardware H.264 encoder max resolution (H.264 Level 4). Larger main streams fail.
_H264_ENCODER_MAX_SIZE = (1920, 1080)

def list_cameras():
    """Return the list of available cameras (from Picamera2.global_camera_info)."""
    camera_info = Picamera2.global_camera_info()
    if not camera_info:
        print("No cameras detected by Picamera2.")
        return None

    print("Available cameras:")
    for index, info in enumerate(camera_info):
        model = info.get("Model", "Unknown")
        location = info.get("Location", "Unknown")
        rotation = info.get("Rotation", "Unknown")
        cam_id = info.get("Id", "Unknown")
        print(
            f"{index}: Model={model}, Location={location}, "
            f"Rotation={rotation}, Id={cam_id}"
        )

    return camera_info


def prompt_for_camera_index(num_cameras: int) -> int:
    """Prompt the user to choose a camera index."""
    while True:
        raw = input(f"Select camera index (0-{num_cameras - 1}): ").strip()
        try:
            index = int(raw)
        except ValueError:
            print("Please enter a valid integer index.")
            continue

        if 0 <= index < num_cameras:
            return index

        print("Index out of range. Try again.")


def list_sensor_modes(camera_index: int):
    """
    Open the given camera, read its sensor modes, print them, and return the list.
    Returns None if the camera has no sensor modes (e.g. not a Pi CSI camera).
    """
    picam2 = Picamera2(camera_index)
    modes = picam2.sensor_modes
    if not modes:
        print("No sensor modes reported for this camera.")
        return None

    print("Available sensor modes:")
    for idx, mode in enumerate(modes):
        size = mode.get("size", (0, 0))
        fps = mode.get("fps", 0)
        bit_depth = mode.get("bit_depth", 0)
        fmt = mode.get("format", "")
        print(
            f"  {idx}: size={size}, fps={fps:.2f}, bit_depth={bit_depth}, format={fmt}"
        )
    picam2.close()
    return modes


def prompt_for_sensor_mode_index(num_modes: int) -> int:
    """Prompt the user to choose a sensor mode index."""
    while True:
        raw = input(f"Select sensor mode index (0-{num_modes - 1}): ").strip()
        try:
            index = int(raw)
        except ValueError:
            print("Please enter a valid integer index.")
            continue

        if 0 <= index < num_modes:
            return index

        print("Index out of range. Try again.")


def prompt_for_action() -> str:
    """Prompt the user to choose between streaming and recording."""
    while True:
        print("Select action:")
        print("  1) Test camera stream (NULL preview for 5 seconds)")
        print("  2) Record 5 second video (MP4)")
        choice = input("Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return choice
        print("Invalid choice. Please enter 1 or 2.")


def run_null_preview(
    camera_index: int,
    duration: int,
    sensor_mode: dict | None,
) -> None:
    """Run the selected camera with a NULL preview for a fixed duration."""
    print(f"Opening camera {camera_index} (NULL preview) for {duration} seconds...")
    picam2 = Picamera2(camera_index)
    if sensor_mode is not None:
        # Request main stream at full sensor size so the pipeline uses full sensor (full FOV).
        # Otherwise the default main size causes a crop and narrower FOV.
        config = picam2.create_preview_configuration(
            main={"size": sensor_mode["size"]},
            sensor={
                "output_size": sensor_mode["size"],
            },
        )
    else:
        config = picam2.create_preview_configuration()
    picam2.configure(config)

    # Start the camera with the NULL preview as documented (no display window).
    picam2.start(show_preview=False)

    try:
        time.sleep(duration)
    finally:
        picam2.stop()

    print("Camera stopped.")


def record_video(
    camera_index: int,
    duration: int,
    filename: str,
    sensor_mode: dict | None,
) -> None:
    """
    Record a fixed-duration video to an MP4 file.

    Picamera2 selects the H.264 encoder and PyavOutput automatically when
    the filename ends in .mp4 (or .ts). The Pi H.264 encoder only supports
    up to 1920x1080; we keep full sensor for FOV and scale main stream to fit.
    """
    print(f"Recording {duration} second video from camera {camera_index} to '{filename}'...")
    picam2 = Picamera2(camera_index)
    if sensor_mode is not None:
        
        ## scale main stream to max encoder size ##
        sensor_size = sensor_mode["size"]
        max_w, max_h = _H264_ENCODER_MAX_SIZE
        w, h = sensor_size
        r = min(max_w / w, max_h / h, 1.0)
        main_size = (int(w * r), int(h * r))
        ###########################################

        config = picam2.create_video_configuration(
            main={"size": main_size},
            sensor={
                "output_size": sensor_size,
            },
        )
    else:
        config = picam2.create_video_configuration()
    picam2.configure(config)
    picam2.start_and_record_video(filename, duration=duration)
    print("Recording complete.")


def main():
    camera_info = list_cameras()
    if not camera_info:
        sys.exit(1)

    index = prompt_for_camera_index(len(camera_info))
    sensor_mode = None
    modes = list_sensor_modes(index)
    if modes:
        mode_index = prompt_for_sensor_mode_index(len(modes))
        sensor_mode = modes[mode_index]

    choice = prompt_for_action()

    if choice == "1":
        run_null_preview(index, 5, sensor_mode)
    else:
        record_video(index, 5, "test.mp4", sensor_mode)

    print("Done. Exiting.")


if __name__ == "__main__":
    main()
    sys.exit(0)

