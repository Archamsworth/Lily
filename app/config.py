"""Configuration loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML config and expand environment variable overrides."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(config_path) as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    # Allow env-var overrides for the most common settings
    lily = cfg.setdefault("lily", {})
    _env_override(lily, "llm.model_path", "LILY_MODEL_PATH")
    _env_override(lily, "server.host", "LILY_HOST")
    _env_override(lily, "server.port", "LILY_PORT", cast=int)
    _env_override(lily, "stt.model", "LILY_WHISPER_MODEL")
    _env_override(lily, "tts.voice", "LILY_PIPER_VOICE")
    return cfg


def _env_override(
    cfg: dict[str, Any],
    dotted_key: str,
    env_var: str,
    cast: type = str,
) -> None:
    value = os.environ.get(env_var)
    if value is None:
        return
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = cast(value)
