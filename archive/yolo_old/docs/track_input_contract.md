# YOLO `track()` Input Contract

This document describes the Ultralytics `YOLO.track()` input contract and the defaults that apply when parameters are omitted.

## Function Signature

```python
model.track(source=None, stream=False, persist=False, **kwargs)
```

- `source`, `stream`, and `persist` are explicit parameters.
- All additional tracking options are passed through `**kwargs`.

## Core Parameters

### `source`

Accepted input types:

- `str` or `Path`: image/video file path, directory path, URL, stream URL (including RTSP).
- `int`: camera device index such as `0`.
- `np.ndarray`: in-memory image/frame.
- `torch.Tensor`: tensor input.
- `list` or `tuple`: batch of sources/images.

Default:

- `None` (Ultralytics may fall back to demo assets internally).

### `stream`

Accepted values:

- `False` (default): process normally and return non-streaming results.
- `True`: return results in streaming mode.

### `persist`

Accepted values:

- `False` (default): tracker state is not persisted across separate calls.
- `True`: tracker state/IDs are persisted across calls on the same model instance.

## Common `**kwargs` Used With Tracking

These are the most relevant options for production tracking pipelines:

- `conf` (`float`): confidence threshold.
- `iou` (`float`): IoU threshold used for NMS.
- `classes` (`int | list[int] | None`): class filter. `None` means all classes.
- `tracker` (`str`): tracker config, commonly `bytetrack.yaml` or `botsort.yaml`, or a custom YAML path.
- `device` (`str | int`): e.g. `"cpu"`, `"cuda:0"`, `0`.
- `imgsz` (`int | list[int]`): inference image size.
- `verbose` (`bool`): control logging verbosity.

Visualization and output flags are also available through `**kwargs`, such as:

- `show`, `save`, `save_txt`, `save_conf`, `show_labels`, `show_conf`, `show_boxes`.

## Tracking-Specific Defaults Applied by Ultralytics

When `track()` is called, Ultralytics sets these values if not explicitly provided:

- `mode = "track"`
- `batch = 1`
- `conf = 0.1` (if no `conf` is passed)

Other parameters are resolved by the predictor/default configuration merge.

## Useful Default Configuration Keys

In Ultralytics `default.yaml`, tracking/predict-related defaults include:

- `iou: 0.7`
- `tracker: botsort.yaml`
- `classes:` (unset by default, which means no class filter)
- `vid_stride: 1`
- `stream_buffer: False`
- `show: False`
- `verbose: True`

These defaults can be overridden at call time via `**kwargs`.
