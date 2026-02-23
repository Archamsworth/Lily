"""Wake-word detection using openwakeword.

Listens to the microphone in a background thread and calls a callback when
the configured wake word is detected.

openwakeword ships with pre-trained models including 'hey_jarvis', 'alexa', etc.
For a custom "hey Lily" wake word a custom model can be trained via the
openwakeword training pipeline and its path placed in config.yaml.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np


class WakeWordDetector:
    """Detects a wake word from the microphone using openwakeword."""

    def __init__(self, cfg: dict[str, Any], on_detected: Callable[[], None]) -> None:
        self._model_name: str = cfg.get("model", "hey_jarvis")
        self._threshold: float = cfg.get("threshold", 0.5)
        self._sample_rate: int = cfg.get("sample_rate", 16000)
        self._frame_size: int = cfg.get("frame_size", 1280)
        self._on_detected = on_detected
        self._running = False
        self._thread: threading.Thread | None = None
        self._oww_model = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start listening in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from openwakeword.model import Model  # type: ignore[import-untyped]
        self._oww_model = Model(
            wakeword_models=[self._model_name],
            inference_framework="tflite",
        )

    def _listen_loop(self) -> None:
        import sounddevice as sd  # type: ignore[import-untyped]

        if self._oww_model is None:
            self._load_model()

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self._frame_size,
        ) as stream:
            while self._running:
                audio_chunk, _ = stream.read(self._frame_size)
                pcm = audio_chunk[:, 0].astype(np.int16)
                predictions = self._oww_model.predict(pcm)
                for score in predictions.values():
                    if score >= self._threshold:
                        # Reset model state to avoid repeated triggers
                        self._oww_model.reset()
                        self._on_detected()
                        break
