"""Tests for the Whisper STT wrapper (uses mocks – no model download required)."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.stt.whisper_stt import WhisperSTT


@pytest.fixture
def mock_stt():
    import app.stt.whisper_stt as stt_module
    mock_whisper = MagicMock()
    mock_model = MagicMock()
    mock_whisper.load_model.return_value = mock_model
    original = stt_module.whisper
    stt_module.whisper = mock_whisper
    yield WhisperSTT({"model": "base", "device": "cpu", "language": "en"}), mock_model
    stt_module.whisper = original


class TestWhisperSTT:
    def test_transcribe_bytes_returns_text(self, mock_stt):
        stt, mock_model = mock_stt
        mock_model.transcribe.return_value = {"text": "  hello world  "}

        # Create a minimal valid WAV in memory
        import io
        import struct
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            silence = struct.pack("<" + "h" * 160, *([0] * 160))
            wf.writeframes(silence)

        result = stt.transcribe_bytes(buf.getvalue())
        assert result == "hello world"

    def test_transcribe_bytes_raw_pcm(self, mock_stt):
        stt, mock_model = mock_stt
        mock_model.transcribe.return_value = {"text": "test pcm"}

        raw_pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
        result = stt.transcribe_bytes(raw_pcm, sample_rate=16000)
        assert result == "test pcm"

    def test_bytes_to_array_produces_float32(self):
        raw_pcm = (np.zeros(1600, dtype=np.int16)).tobytes()
        arr = WhisperSTT._bytes_to_array(raw_pcm, 16000)
        assert arr.dtype == np.float32
        assert arr.ndim == 1

    def test_bytes_to_array_resamples(self):
        # 8 kHz input → 16 kHz
        raw_pcm = (np.zeros(800, dtype=np.int16)).tobytes()
        arr = WhisperSTT._bytes_to_array(raw_pcm, 8000)
        assert len(arr) == 1600
