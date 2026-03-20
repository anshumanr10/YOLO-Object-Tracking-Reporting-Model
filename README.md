# YOLO Object Tracking + WebRTC Stream Server

FastAPI + WebRTC server for Raspberry Pi camera streaming with two browser modes:

- **Raw stream** (camera only)
- **Tracking stream** (camera + YOLO object detection/tracking overlays)

The app serves frontend pages and negotiates WebRTC in the backend, then sends frames from aiortc `VideoStreamTrack` implementations.

## Current Feature Set

- Live WebRTC video stream in browser (`/` and `/tracking`)
- Pi camera discovery and sensor mode listing
- Per-session stream options via cookie-based session state
- Runtime tracking controls from UI:
  - model selection
  - target FPS
  - confidence threshold
  - class filter (multi-select)
  - camera and sensor mode
- Optional model preload endpoint for faster first stream start

## High-Level Architecture

```
Browser page (/, /tracking)
  -> WebSocket (/ws or /ws-tracking)
  -> stream_fastapi.routes.websocket
  -> webrtc.handle_webrtc_connection(...)
  -> VideoStreamTrack
     - Picamera2Source (raw)
     - TrackingVideoTrack (YOLO tracking)
```

### Stream Modes

- **Raw mode**
  - Page: `/`
  - WebSocket: `/ws`
  - Track: `pi.picamera2_source.Picamera2Source`

- **Tracking mode**
  - Page: `/tracking`
  - WebSocket: `/ws-tracking`
  - Track: `pi.tracking_source.TrackingVideoTrack`
  - Tracking core: `yolo.tracker.tracking_frames(...)`

> Single-camera system: run one active stream at a time.

## Project Layout

- `main.py` - ASGI entrypoint (`app` imported from `stream_fastapi`)
- `stream_fastapi/` - FastAPI app factory and routes
- `frontend/` - static HTML/CSS/JS for viewer pages
- `webrtc/` - reusable WebRTC signaling/connection flow
- `pi/` - aiortc video tracks and camera helpers
- `yolo/` - model config loading, frame sources, tracker pipeline
- `docs/` - development notes and context docs
- `archived/` - older prototypes/reference code

## Requirements

### Hardware / OS

- Raspberry Pi with camera support (Picamera2/libcamera)
- Linux environment with camera access enabled

### Python

- Python 3.13 environment (as currently used in this repo)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Start server:

```bash
python main.py
```

Alternative:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open in browser:

- Raw stream: `http://<host>:8000/`
- Tracking stream: `http://<host>:8000/tracking`

## HTTP + WebSocket Endpoints

### Pages

- `GET /` -> raw stream viewer
- `GET /tracking` -> tracking stream viewer

### WebSocket signaling

- `WS /ws` -> raw stream WebRTC signaling
- `WS /ws-tracking` -> tracking stream WebRTC signaling

### Camera metadata + raw stream options

- `GET /api/cameras`
- `GET /api/cameras/{camera_index}/modes`
- `GET /api/stream/options`
- `PATCH /api/stream/options`

### Tracking controls

- `GET /tracking/load-model` (model preload)
- `GET /tracking/warmup` (legacy alias)
- `GET /tracking/options`
- `PATCH /tracking/options`

## Tracking Configuration

Tracking defaults are loaded from:

- `yolo/config/defaults.yaml`
- `yolo/config/models.yaml`
- `yolo/config/classifications.yaml`

Primary defaults in `defaults.yaml`:

- `source.type` (for example `PiCamera`)
- `model` (for example `yolov8n`)
- `conf`
- `fps`
- `tracking`
- `classes`

Model keys (from `models.yaml`) currently include:

- `yolov8n`
- `yolov8s`
- `yolov8m`
- `yolov8l`
- `yolov8x`

## Session Behavior

- Session ID is stored in a `session_id` cookie.
- Each session keeps:
  - one raw `Picamera2Source` track
  - one tracking `TrackingVideoTrack` track
- Options APIs update the session-specific track settings.

## Frontend Notes

- Shared client script: `frontend/js/stream.js`
- Raw page uses `/ws` and `/api/stream/options`
- Tracking page uses `/ws-tracking` and `/tracking/options`
- Tracking page calls `/tracking/load-model` before enabling Start

## Quick Troubleshooting

- **No cameras listed**
  - Verify camera is connected and enabled in Pi camera stack.
  - Check Picamera2 install and permissions.
- **WebSocket closes immediately**
  - Check server logs for exceptions from track startup.
- **Tracking starts slowly on first run**
  - Use `/tracking/load-model` (already called by tracking page) to preload model.
- **Choppy stream**
  - Lower FPS and/or choose a lighter model (`yolov8n`).

## Utility Scripts

- `pi/test_picamera.py` - minimal camera open/capture test
- `pi/test_picamera_options.py` - interactive camera/mode test
