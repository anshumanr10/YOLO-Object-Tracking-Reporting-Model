from pathlib import Path
import cv2
from . import config_loader as config

#For testing over WSL-Windows Tunnel
from old.get_windows_host_ip import get_windows_host_ip

def load_video_source(video_source: dict | None = None) -> cv2.VideoCapture:
    if video_source is None:
        config.load_config()
        src = config.defaults["source"]          
    else:
         src = video_source
    
    src_type = src["type"]
    src_vals = src["values"]
    if src_type == "Webcam":
        cap = cv2.VideoCapture(src_vals["int"])
    elif src_type in {"RTMP", "RTSP", "HTML"}:
        cap = cv2.VideoCapture(src_vals["url"])
    elif src_type == "File":
        cap = cv2.VideoCapture(src_vals["path"])
    else:
        raise ValueError(f"Unsupported source type: {src_type}")
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: type={src_type}, values={src_vals}")
    return cap

if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    
    ip = get_windows_host_ip()
    cap = load_video_source({
    "type": "RTSP",
    "values": {"url": f"rtsp://{ip}:8554/live"}
    })
    print("Successfully connected to video source.")
    cap.release()