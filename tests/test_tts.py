"""Tests for PiperTTS (uses mocks – piper binary not required)."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from app.tts.piper_tts import PiperTTS, _strip_expressions


class TestStripExpressions:
    def test_strips_single_marker(self):
        assert _strip_expressions("*smiles* Hello!") == "Hello!"

    def test_strips_multiple_markers(self):
        result = _strip_expressions("*pouts* Where were you? *folds arms*")
        assert result == "Where were you?"

    def test_no_markers_unchanged(self):
        assert _strip_expressions("Just text.") == "Just text."

    def test_empty_string(self):
        assert _strip_expressions("") == ""


class TestPiperTTS:
    @pytest.fixture
    def tts(self, tmp_path):
        voices_dir = tmp_path / "voices"
        voices_dir.mkdir()
        # Create a dummy onnx voice file so the path check passes
        (voices_dir / "en_US-lessac-medium.onnx").write_bytes(b"dummy")
        return PiperTTS({
            "voice": "en_US-lessac-medium",
            "speed": 1.0,
            "piper_executable": "piper",
            "voices_dir": str(voices_dir),
        })

    @patch("app.tts.piper_tts.subprocess.run")
    @patch("app.tts.piper_tts.Path")
    def test_synthesize_calls_piper(self, mock_path_cls, mock_run, tts):
        # Simulate piper writing a WAV file
        mock_path_inst = MagicMock()
        mock_path_inst.read_bytes.return_value = b"RIFF" + b"\x00" * 40
        mock_path_inst.unlink = MagicMock()

        # The Path calls we care about: tmp file read and unlink
        mock_path_cls.side_effect = lambda p: mock_path_inst if p.endswith(".wav") else Path(p)

        mock_run.return_value = MagicMock(returncode=0)

        result = tts.synthesize("Hello world")
        assert mock_run.called

    def test_empty_text_returns_empty_bytes(self, tts):
        result = tts.synthesize("*smiles*")  # only expression, no spoken text
        assert result == b""

    def test_synthesize_stream_yields_chunks(self, tts):
        with patch.object(tts, "synthesize", return_value=b"A" * 100):
            chunks = list(tts.synthesize_stream("Hello", chunk_size=32))
        assert sum(len(c) for c in chunks) == 100
        assert all(len(c) <= 32 for c in chunks)
