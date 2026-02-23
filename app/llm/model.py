"""Local LLM wrapper using llama-cpp-python.

Loads a GGUF model (default: qwen2.5-3b-instruct-q4_k_m.gguf) and exposes a
streaming chat interface used by the FastAPI server.
"""
from __future__ import annotations

import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama  # type: ignore[import-untyped]
except ImportError:
    Llama = None  # type: ignore[assignment,misc]


class LilyLLM:
    """Wrapper around a GGUF model loaded with llama-cpp-python."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        model_path = Path(cfg.get("model_path", "models/qwen2.5-3b-instruct-q4_k_m.gguf"))
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Place the GGUF file there or update 'llm.model_path' in config.yaml."
            )
        self._max_tokens: int = cfg.get("max_tokens", 512)
        self._temperature: float = cfg.get("temperature", 0.7)
        self._top_p: float = cfg.get("top_p", 0.9)
        self._top_k: int = cfg.get("top_k", 40)
        self._repeat_penalty: float = cfg.get("repeat_penalty", 1.1)
        self._system_prompt: str = cfg.get(
            "system_prompt",
            "You are Lily, a warm and expressive virtual companion.",
        )

        if Llama is None:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
            )
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=cfg.get("context_length", 4096),
            n_gpu_layers=cfg.get("n_gpu_layers", 0),
            n_threads=cfg.get("n_threads", 4),
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        user_message: str,
        context: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> Generator[str, None, None]:
        """Yield raw tokens (including *expression* markers) as they are generated."""
        messages = self._build_messages(user_message, context, history or [])
        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            top_k=self._top_k,
            repeat_penalty=self._repeat_penalty,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token

    def chat(
        self,
        user_message: str,
        context: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Return full response as a single string."""
        return "".join(self.chat_stream(user_message, context, history))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        user_message: str,
        context: str,
        history: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        system_content = self._system_prompt
        if context:
            system_content += (
                "\n\nRelevant web context (use if helpful, cite sources when possible):\n"
                + context
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": user_message})
        return messages

    @staticmethod
    def strip_expressions(text: str) -> str:
        """Remove *expression* markers, returning only spoken text."""
        cleaned = re.sub(r"\*[^*]+\*", "", text)
        return re.sub(r" {2,}", " ", cleaned).strip()
