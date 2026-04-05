from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml


YamlPath = Union[str, Path]


def _default_config_path() -> Path:
    # Resolve relative to this package so it works regardless of CWD.
    return Path(__file__).resolve().parent / "tracking_defaults.yaml"


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    """
    Load YAML whose root is expected to be a mapping.

    Contract for this project:
    - If YAML parses to None (e.g., comment-only file), treat as {}.
    - No validation of keys/values; caller is responsible.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if isinstance(data, dict):
        return dict(data)
    raise ValueError(f"YAML root must be a mapping (dict) or empty. Got {type(data)} in {path}")


@dataclass
class ConfigManager:
    """
    Stateful configuration store:
    - defaults: loaded once from YAML on initialization (immutable snapshot)
    - overrides: user-provided updates (mutable)

    No validation is performed. Invalid keys/values are passed through to the
    underlying consumer (Ultralytics) and are the caller's responsibility.
    """

    config_path: Path = field(default_factory=_default_config_path)
    _defaults: Dict[str, Any] = field(init=False, repr=False)
    _overrides: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)
        self._defaults = _load_yaml_mapping(self.config_path)

    @property
    def defaults(self) -> Mapping[str, Any]:
        return dict(self._defaults)

    @property
    def overrides(self) -> Mapping[str, Any]:
        return dict(self._overrides)

    def set_config(self, patch: Mapping[str, Any]) -> None:
        self._overrides.update(dict(patch))

    def reset_config(self) -> None:
        self._overrides.clear()

    def get_config(self, *, include_defaults: bool = True) -> Dict[str, Any]:
        if not include_defaults:
            return dict(self._overrides)
        merged = dict(self._defaults)
        merged.update(self._overrides)
        return merged

    def effective_kwargs(self, call_kwargs: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        merged = self.get_config(include_defaults=True)
        if call_kwargs:
            merged.update(dict(call_kwargs))
        return merged

