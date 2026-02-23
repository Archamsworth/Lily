"""Speech-to-text using OpenAI Whisper (offline).

Accepts raw PCM bytes (16-bit, 16 kHz, mono) or a WAV file path and returns
the transcribed text.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
try:
    import whisper  # type: ignore[import-untyped]
except ImportError:
    whisper = None  # type: ignore[assignment]


class WhisperSTT:
    """Thin wrapper around the Whisper model."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        if whisper is None:
            raise ImportError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )
        model_name: str = cfg.get("model", "base")
        device: str = cfg.get("device", "cpu")
        self._language: str | None = cfg.get("language") or None
        self._model = whisper.load_model(model_name, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe an audio file (any format Whisper supports)."""
        result = self._model.transcribe(
            str(path),
            language=self._language,
            fp16=False,
        )
        return result["text"].strip()

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe raw PCM bytes (16-bit signed, mono) or a WAV blob.

        Whisper expects a float32 NumPy array sampled at 16 kHz.
        """
        audio_array = self._bytes_to_array(audio_bytes, sample_rate)
        result = self._model.transcribe(
            audio_array,
            language=self._language,
            fp16=False,
        )
        return result["text"].strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bytes_to_array(data: bytes, sample_rate: int) -> np.ndarray:
        """Convert raw bytes to a float32 numpy array at 16 kHz."""
        try:
            # Try reading as a soundfile (WAV / FLAC / OGG)
            buf = io.BytesIO(data)
            array, sr = sf.read(buf, dtype="float32")
            if sr != 16000:
                # Resample using numpy linear interpolation (lightweight)
                duration = len(array) / sr
                new_len = int(duration * 16000)
                array = np.interp(
                    np.linspace(0, len(array) - 1, new_len),
                    np.arange(len(array)),
                    array,
                )
        except Exception:
            # Fallback: assume raw 16-bit PCM at the given sample_rate
            array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if sample_rate != 16000:
                duration = len(array) / sample_rate
                new_len = int(duration * 16000)
                array = np.interp(
                    np.linspace(0, len(array) - 1, new_len),
                    np.arange(len(array)),
                    array,
                )
        # Ensure mono
        if array.ndim > 1:
            array = array.mean(axis=1)
        return array.astype(np.float32)
