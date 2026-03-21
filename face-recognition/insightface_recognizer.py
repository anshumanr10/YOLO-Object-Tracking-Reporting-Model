#!/usr/bin/env python3
"""
Live face detection + recognition using InsightFace.

- Builds a small face gallery from images in ./known_faces by default.
- Runs detection/recognition on RTSP or webcam feed.
- Draws bbox + recognized identity on each frame.
"""

from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GALLERY_DIR = SCRIPT_DIR / "known_faces"
MATCH_THRESHOLD = 0.45


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(vec)
    if denom <= 1e-12:
        return vec
    return vec / denom


def _collect_image_paths(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def build_gallery(app: FaceAnalysis, gallery_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
    Build known identities from image files in gallery_dir.
    Naming convention: one identity per file, name from file stem.
    """
    if not gallery_dir.is_dir():
        raise RuntimeError(f"Gallery folder not found: {gallery_dir}")

    image_paths = _collect_image_paths(gallery_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in gallery folder: {gallery_dir}")

    embeddings: list[np.ndarray] = []
    names: list[str] = []

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipping unreadable image: {image_path.name}")
            continue

        faces = app.get(img)
        if not faces:
            print(f"Skipping (no face): {image_path.name}")
            continue

        # If multiple faces are present, use the highest detection score.
        best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
        emb = _l2_normalize(np.asarray(best_face.embedding, dtype=np.float32))

        embeddings.append(emb)
        names.append(image_path.stem)

    if not embeddings:
        raise RuntimeError("No valid faces found in gallery images.")

    return np.vstack(embeddings), names


def recognize_face(
    query_embedding: np.ndarray, gallery_embeddings: np.ndarray, gallery_names: list[str]
) -> tuple[str, float]:
    query = _l2_normalize(query_embedding.astype(np.float32))
    scores = gallery_embeddings @ query
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    name = gallery_names[best_idx] if best_score >= MATCH_THRESHOLD else "unknown"
    return name, best_score


def _open_source(raw: str) -> cv2.VideoCapture:
    source: int | str
    source = int(raw) if raw.isdigit() else raw
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG if isinstance(source, str) else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main() -> None:
    source = input("Video source (RTSP URL or webcam index, e.g. 0): ").strip()
    if not source:
        print("No source given.")
        return

    gallery_input = input(f"Gallery folder [{DEFAULT_GALLERY_DIR}]: ").strip()
    gallery_dir = Path(gallery_input).expanduser() if gallery_input else DEFAULT_GALLERY_DIR

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    try:
        gallery_embeddings, gallery_names = build_gallery(app, gallery_dir)
    except RuntimeError as e:
        print(e)
        print("Add at least one face image to the gallery folder and retry.")
        return

    cap = _open_source(source)
    if not cap.isOpened():
        print("Could not open video source.")
        return

    window = "InsightFace recognition (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Frame read failed; exiting.")
                break

            faces = app.get(frame)
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                name, score = recognize_face(face.embedding, gallery_embeddings, gallery_names)
                label = f"{name} ({score:.2f})"
                color = (0, 200, 0) if name != "unknown" else (0, 140, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
