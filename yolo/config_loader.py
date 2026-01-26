from pathlib import Path
from typing import NamedTuple, Dict, Any
import yaml

CONFIG_DIR = Path("config")
_LOADED = False
defaults: dict = {}
video_sources: dict = {}
models: dict = {}
classifications: dict = {}

# GLOBAL MODULE: access data via:
#   import config_loader as config;
#   config.defaults; config.models; ...
def load_config(config_dir: Path = CONFIG_DIR) -> None:
    global _LOADED, defaults, video_sources, models, classifications

    if _LOADED:
        return
    
    defaults = _load_yaml(CONFIG_DIR / "defaults.yaml")
    video_sources = _load_yaml(CONFIG_DIR / "video_sources.yaml")
    models = _load_yaml(CONFIG_DIR / "models.yaml")
    classifications = _load_yaml(CONFIG_DIR / "classifications.yaml")

    _validate_defaults(defaults)
    _validate_video_sources(video_sources)
    _validate_models(models)
    _validate_classifications(classifications)
    _LOADED = True

def reset_config() -> None:
    global _LOADED, defaults, video_sources, models, classifications
    _LOADED = False
    defaults = {}
    video_sources = {}
    models = {}
    classifications = {}

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping (dict). Current: {type(data)} in {path}")
    return data

def _validate_defaults(yaml_dict: Dict[str, Any]) -> None:
    if not isinstance(yaml_dict, dict):
        raise ValueError(f"defaults.yaml root must be a dict, got {type(yaml_dict)}")

    # required: model (str)
    model = yaml_dict.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("defaults.yaml: missing/invalid 'model' (expected non-empty string)")

    # required: source (dict) with type (str)
    source = yaml_dict.get("source")
    if not isinstance(source, dict):
        raise ValueError("defaults.yaml: missing/invalid 'source' (expected mapping)")
    stype = source.get("type")
    if not isinstance(stype, str) or not stype.strip():
        raise ValueError("defaults.yaml: missing/invalid 'source.type' (expected non-empty string)")

    # optional numeric fields (accept int/float)
    for k in ("conf", "fps"):
        if k in yaml_dict and yaml_dict[k] is not None and not isinstance(yaml_dict[k], (int, float)):
            raise ValueError(f"defaults.yaml: invalid '{k}' (expected number), got {type(yaml_dict[k])}")

    # optional: tracking (bool or mapping)
    if "tracking" in yaml_dict and yaml_dict["tracking"] is not None:
        tracking = yaml_dict["tracking"]
        if not isinstance(tracking, (bool, dict)):
            raise ValueError(
                f"defaults.yaml: invalid 'tracking' (expected bool or mapping), got {type(tracking)}"
            )

def _validate_video_sources(yaml_dict: Dict[str, Any]) -> None:
    if not isinstance(yaml_dict, dict):
        raise ValueError(f"input_sources.yaml root must be a dict, got {type(yaml_dict)}")

    if not yaml_dict:
        raise ValueError("input_sources.yaml is empty")

    for source_name, spec in yaml_dict.items():
        if not isinstance(source_name, str) or not source_name.strip():
            raise ValueError(f"input_sources.yaml: invalid source key {source_name!r}")

        if not isinstance(spec, dict):
            raise ValueError(f"input_sources.yaml: '{source_name}' must map to a dict, got {type(spec)}")

        label = spec.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"input_sources.yaml: '{source_name}.label' missing/invalid")

        keys = {"url", "index", "path"} & spec.keys()
        if len(keys) != 1:
            raise ValueError("video source must be exactly one of {url/index/path}")

def _validate_models(yaml_dict: Dict[str, Any]) -> None:
    if not isinstance(yaml_dict, dict):
        raise ValueError(f"models.yaml root must be a dict, got {type(yaml_dict)}")
    if not yaml_dict:
        raise ValueError("models.yaml is empty")

    for model_key, spec in yaml_dict.items():
        if not isinstance(model_key, str) or not model_key.strip():
            raise ValueError(f"models.yaml: invalid model key {model_key!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"models.yaml: '{model_key}' must map to a dict, got {type(spec)}")

        weights = spec.get("weights")
        if not isinstance(weights, str) or not weights.strip():
            raise ValueError(f"models.yaml: '{model_key}.weights' missing/invalid")

        task = spec.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"models.yaml: '{model_key}.task' missing/invalid")

        size = spec.get("size")
        if not isinstance(size, str) or not size.strip():
            raise ValueError(f"models.yaml: '{model_key}.size' missing/invalid")

def _validate_classifications(yaml_dict: Dict[str, Any]) -> None:
    if not isinstance(yaml_dict, dict):
        raise ValueError(f"classifications.yaml root must be a dict, got {type(yaml_dict)}")
    if not yaml_dict:
        raise ValueError("classifications.yaml is empty")

    ids = []
    for name, cid in yaml_dict.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"classifications.yaml: invalid class name key {name!r}")
        if not isinstance(cid, int):
            raise ValueError(f"classifications.yaml: '{name}' must map to an int id, got {type(cid)}")
        if cid < 0:
            raise ValueError(f"classifications.yaml: '{name}' has negative id {cid}")
        ids.append(cid)

    if len(set(ids)) != len(ids):
        raise ValueError("classifications.yaml: duplicate class IDs found")

    # COCO sanity (your file is 0..79)
    if min(ids) != 0 or max(ids) != 79 or len(ids) != 80:
        raise ValueError(
            f"classifications.yaml: expected 80 classes with ids 0..79; got count={len(ids)}, min={min(ids)}, max={max(ids)}"
        )

if __name__ == "__main__":
    print(f"Running: {Path(__file__).resolve()}")
    cfg = load_config()
    print("All YAML files loaded and validated.")