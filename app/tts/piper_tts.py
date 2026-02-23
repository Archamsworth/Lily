"""Text-to-speech using Piper TTS.

Piper is called as a subprocess: it reads text on stdin and writes a WAV file.
The WAV bytes are returned so the server can stream them to the client.

Piper binary install:  pip install piper-tts
Voice models are placed in the ``voices/`` directory; download them from
https://huggingface.co/rhasspy/piper-voices.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class PiperTTS:
    """Wrapper around the Piper TTS binary."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._voice: str = cfg.get("voice", "en_US-lessac-medium")
        self._speed: float = cfg.get("speed", 1.0)
        self._piper_bin: str = cfg.get("piper_executable", "piper")
        self._voices_dir: Path = Path(cfg.get("voices_dir", "voices"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> bytes:
        """Convert *text* to WAV audio bytes using Piper.

        Strips *expression* markers before synthesis so they are not spoken.
        """
        clean_text = _strip_expressions(text)
        if not clean_text:
            return b""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        model_path = self._voices_dir / f"{self._voice}.onnx"
        cmd = [
            self._piper_bin,
            "--model", str(model_path),
            "--output_file", tmp_path,
            "--length_scale", str(1.0 / max(self._speed, 0.1)),
        ]
        try:
            subprocess.run(
                cmd,
                input=clean_text.encode("utf-8"),
                check=True,
                capture_output=True,
                timeout=30,
            )
            return Path(tmp_path).read_bytes()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def synthesize_stream(self, text: str, chunk_size: int = 4096):
        """Yield WAV audio in *chunk_size* byte chunks."""
        data = self.synthesize(text)
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_expressions(text: str) -> str:
    """Remove *expression* markers from text before TTS synthesis."""
    import re
    return re.sub(r"\*[^*]+\*", "", text).strip()
