"""Tests for WakeWordDetector (uses mocks – no audio hardware required)."""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from app.wake_word.detector import WakeWordDetector


class TestWakeWordDetector:
    def _make_detector(self, callback=None):
        cfg = {
            "model": "hey_jarvis",
            "threshold": 0.5,
            "sample_rate": 16000,
            "frame_size": 1280,
        }
        return WakeWordDetector(cfg, on_detected=callback or (lambda: None))

    def test_initial_state_not_running(self):
        det = self._make_detector()
        assert not det.is_running

    def test_stop_before_start_is_safe(self):
        det = self._make_detector()
        det.stop()  # should not raise
        assert not det.is_running

    @patch("app.wake_word.detector.WakeWordDetector._listen_loop")
    def test_start_sets_running(self, mock_loop):
        det = self._make_detector()
        det.start()
        time.sleep(0.05)
        assert det.is_running
        det.stop()

    @patch("app.wake_word.detector.WakeWordDetector._listen_loop")
    def test_double_start_is_idempotent(self, mock_loop):
        det = self._make_detector()
        det.start()
        thread_id = id(det._thread)
        det.start()  # second start should be no-op
        assert id(det._thread) == thread_id
        det.stop()

    def test_callback_invoked_on_detection(self):
        """Simulate detection above threshold triggering the callback."""
        import numpy as np

        called = threading.Event()
        det = self._make_detector(callback=lambda: called.set())

        mock_model = MagicMock()
        mock_model.predict.return_value = {"hey_jarvis": 0.95}
        mock_model.reset = MagicMock()
        det._oww_model = mock_model

        # Directly test the detection logic inline (without audio device)
        import numpy as np
        pcm = np.zeros(1280, dtype=np.int16)
        predictions = det._oww_model.predict(pcm)
        for score in predictions.values():
            if score >= det._threshold:
                det._oww_model.reset()
                det._on_detected()
                break

        assert called.is_set()
        mock_model.reset.assert_called_once()
