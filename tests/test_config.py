"""Tests for the configuration loader."""
import os
from pathlib import Path

import pytest
import yaml

from app.config import load_config


@pytest.fixture
def config_file(tmp_path):
    cfg = {
        "lily": {
            "llm": {"model_path": "models/test.gguf", "max_tokens": 256},
            "server": {"host": "0.0.0.0", "port": 8080},
            "stt": {"model": "tiny"},
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return str(path)


class TestLoadConfig:
    def test_loads_yaml_correctly(self, config_file):
        cfg = load_config(config_file)
        assert cfg["lily"]["llm"]["model_path"] == "models/test.gguf"
        assert cfg["lily"]["server"]["port"] == 8080

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_env_var_overrides_model_path(self, config_file, monkeypatch):
        monkeypatch.setenv("LILY_MODEL_PATH", "/override/model.gguf")
        cfg = load_config(config_file)
        assert cfg["lily"]["llm"]["model_path"] == "/override/model.gguf"

    def test_env_var_overrides_port_as_int(self, config_file, monkeypatch):
        monkeypatch.setenv("LILY_PORT", "9000")
        cfg = load_config(config_file)
        assert cfg["lily"]["server"]["port"] == 9000
        assert isinstance(cfg["lily"]["server"]["port"], int)

    def test_env_var_overrides_host(self, config_file, monkeypatch):
        monkeypatch.setenv("LILY_HOST", "0.0.0.0")
        cfg = load_config(config_file)
        assert cfg["lily"]["server"]["host"] == "0.0.0.0"
