# YOLO Object Tracking & Reporting Model

FastAPI backend that streams live video from a Raspberry Pi camera over WebRTC. Two modes: raw camera stream or camera + YOLO object tracking with optional reporting.

## Pipeline

```
Frontend (browser)
    │  loads pages (e.g. /, /tracking), connects WebSocket (/ws or /ws-tracking)
    ▼
stream_fastapi  (routes: pages, /ws, /ws-tracking)
    │  on WebSocket connect: calls handle_webrtc_connection(..., track_factory=...)
    ▼
webrtc  (handle_webrtc_connection)
    │  runs signaling, then streams the track from the factory
    ▼
pi/picamera2_source.Picamera2Source  OR  pi/tracking_source.TrackingVideoTrack
    │  Picamera2Source: capture loop → queue → recv() → VideoFrame
    │  TrackingVideoTrack: camera stream → tracking_frames() → queue → recv() → VideoFrame
    ▼
yolo  (only for TrackingVideoTrack: config_loader, tracker.tracking_frames, etc.)
```

- **Raw stream:** `/` → WebSocket `/ws` → `Picamera2Source` (camera only).
- **Tracking stream:** `/tracking` → WebSocket `/ws-tracking` → `TrackingVideoTrack` (camera + YOLO tracking). Uses `yolo` for config, model, and tracking.

Only one stream at a time (single camera).

## Run

```bash
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000
```

Open `http://<host>:8000` for raw camera or `http://<host>:8000/tracking` for tracking.

## Layout

- **stream_fastapi/** — FastAPI app, routes, WebSocket endpoints.
- **webrtc/** — WebRTC signaling and connection (aiortc); no app-specific code.
- **pi/** — Video track implementations: `Picamera2Source`, `TrackingVideoTrack` (backend only).
- **yolo/** — YOLO config, tracker, frame sources, video output, reporting; no WebRTC/FastAPI.

Config: `yolo/config/defaults.yaml` (model, source, conf, fps, tracking, classes).
