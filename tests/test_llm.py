"""Tests for the LLM model wrapper (uses mocks – no GGUF file required)."""
import app.llm.model as llm_module
from unittest.mock import MagicMock, patch

import pytest

from app.llm.model import LilyLLM


@pytest.fixture
def mock_llm_cfg(tmp_path):
    """Config dict pointing to a dummy model file."""
    model_file = tmp_path / "dummy.gguf"
    model_file.write_bytes(b"dummy")
    return {
        "model_path": str(model_file),
        "context_length": 512,
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "n_gpu_layers": 0,
        "n_threads": 1,
        "system_prompt": "You are Lily.",
    }


class TestLilyLLM:
    def test_missing_model_raises(self):
        with pytest.raises(FileNotFoundError):
            LilyLLM({"model_path": "/nonexistent/model.gguf"})

    def test_missing_llama_raises(self, mock_llm_cfg):
        original = llm_module.Llama
        llm_module.Llama = None
        try:
            with pytest.raises(ImportError):
                LilyLLM(mock_llm_cfg)
        finally:
            llm_module.Llama = original

    @patch("app.llm.model.Llama")
    def test_chat_returns_string(self, mock_llama_cls, mock_llm_cfg):
        mock_instance = MagicMock()
        mock_instance.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": "!"}}]},
            {"choices": [{"delta": {}}]},
        ])
        mock_llama_cls.return_value = mock_instance

        llm = LilyLLM(mock_llm_cfg)
        result = llm.chat("Hi")
        assert result == "Hello!"

    @patch("app.llm.model.Llama")
    def test_chat_stream_yields_tokens(self, mock_llama_cls, mock_llm_cfg):
        mock_instance = MagicMock()
        mock_instance.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
        ])
        mock_llama_cls.return_value = mock_instance

        llm = LilyLLM(mock_llm_cfg)
        tokens = list(llm.chat_stream("Test"))
        assert tokens == ["A", "B"]

    def test_strip_expressions(self):
        text = "*smiles* Hello *waves* world"
        assert LilyLLM.strip_expressions(text) == "Hello world"

    def test_strip_expressions_no_markers(self):
        text = "Hello world"
        assert LilyLLM.strip_expressions(text) == "Hello world"

    @patch("app.llm.model.Llama")
    def test_history_included_in_messages(self, mock_llama_cls, mock_llm_cfg):
        mock_instance = MagicMock()
        mock_instance.create_chat_completion.return_value = iter([])
        mock_llama_cls.return_value = mock_instance

        llm = LilyLLM(mock_llm_cfg)
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        list(llm.chat_stream("What's up?", history=history))

        call_kwargs = mock_instance.create_chat_completion.call_args
        messages = call_kwargs[1]["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]
