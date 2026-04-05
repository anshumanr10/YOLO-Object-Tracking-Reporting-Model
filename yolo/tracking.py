from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable, Mapping, Optional, Tuple, Union

from .config_manager import ConfigManager, YamlPath


ModelSpec = Union[str, Path]

if TYPE_CHECKING:
    from ultralytics import YOLO  # pragma: no cover


class YOLOTracker:
    """
    Thin, stateful wrapper around Ultralytics' YOLO.track() for frame-by-frame use.

    Contract:
    - On init, loads defaults from tracking_defaults.yaml (comment-only YAML -> {}).
    - Users may call set_config() any time to update overrides (no validation).
    - track_frame() can be called repeatedly; it uses defaults < overrides < per-call kwargs.
    - source is always the provided frame; any source in config/kwargs is ignored.
    - stream is always False for per-frame ``Results`` (list); ``stream`` in config/kwargs is ignored.
    - No persistence: restart = reload YAML again.
    """

    def __init__(
        self,
        model: Union["YOLO", ModelSpec],
        *,
        config_path: Optional[YamlPath] = None,
    ) -> None:
        self.config = ConfigManager(config_path=config_path or (Path(__file__).resolve().parent / "tracking_defaults.yaml"))
        try:
            from ultralytics import YOLO  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ultralytics is required to use YOLOTracker. Install it (e.g. `pip install ultralytics`)."
            ) from e

        self.model: "YOLO" = model if isinstance(model, YOLO) else YOLO(model)

    def get_config(self) -> Dict[str, Any]:
        return self.config.get_config(include_defaults=True)

    def set_config(self, patch: Mapping[str, Any]) -> None:
        self.config.set_config(patch)

    def reset_config(self) -> None:
        self.config.reset_config()

    def track_frame(self, frame: Any, **kwargs: Any) -> Any:
        effective = self.config.effective_kwargs(kwargs)
        effective.pop("source", None)
        effective.pop("stream", None)
        effective["stream"] = False
        return self.model.track(source=frame, **effective)


def track_loop(
    tracker: YOLOTracker,
    frame_stream: Iterable[Any],
    **track_kwargs: Any,
) -> Generator[Any, None, None]:
    """
    Pull-based tracking loop.

    Consumes frames from a standard iterator/generator and yields the Ultralytics
    `Results` output for each processed frame.
    """
    for frame in frame_stream:
        yield tracker.track_frame(frame, **track_kwargs)


LatestFrameGetter = Callable[[], Optional[Tuple[Any, int, float]]]


def track_latest(
    tracker: YOLOTracker,
    get_latest_frame: LatestFrameGetter,
    **track_kwargs: Any,
) -> Generator[Any, None, None]:
    """
    Latest-only tracking loop (skeleton).

    Intended future shape for a background capture producer:
    - `get_latest_frame()` returns (frame_ref, frame_id, timestamp) or None.
    - This loop would poll until `frame_id` changes, then run inference on that
      newest frame. Intermediate frames are implicitly dropped.
    """
    raise NotImplementedError("Use track_loop() for pull-based sources for now.")
