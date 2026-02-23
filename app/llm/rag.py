"""RAG (Retrieval-Augmented Generation) helper.

Uses DuckDuckGo for web search and BeautifulSoup for extracting page text so
that Lily can answer questions about current events without an API key.
"""
from __future__ import annotations

import re
from typing import Any

import requests
try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]

try:
    from duckduckgo_search import DDGS  # type: ignore[import-untyped]
except ImportError:
    DDGS = None  # type: ignore[assignment,misc]


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


class RAGRetriever:
    """Retrieves web context for a user query."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._enabled: bool = cfg.get("enabled", True)
        self._num_results: int = cfg.get("search_results", 3)
        self._chunk_size: int = cfg.get("chunk_size", 512)
        self._max_context_chars: int = cfg.get("max_context_chars", 2048)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context(self, query: str) -> str:
        """Return a condensed web context string for the query, or '' if disabled."""
        if not self._enabled:
            return ""
        results = self._search(query)
        if not results:
            return ""
        chunks: list[str] = []
        total = 0
        for r in results:
            snippet = self._fetch_snippet(r["href"], r.get("body", ""))
            if not snippet:
                continue
            source_line = f"[{r['title']}]({r['href']})"
            entry = f"{source_line}\n{snippet}"
            entry_len = len(entry)
            if total + entry_len > self._max_context_chars:
                remaining = self._max_context_chars - total
                if remaining > 100:
                    chunks.append(entry[:remaining])
                break
            chunks.append(entry)
            total += entry_len
        return "\n\n".join(chunks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _search(self, query: str) -> list[dict[str, str]]:
        if DDGS is None:
            return []
        try:
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=self._num_results))
        except Exception:
            return []

    def _fetch_snippet(self, url: str, fallback: str) -> str:
        if BeautifulSoup is None:
            return fallback[: self._chunk_size]
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            # Remove scripts / styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s{2,}", " ", text)
            return text[: self._chunk_size]
        except Exception:
            return fallback[: self._chunk_size]

    @staticmethod
    def should_search(query: str) -> bool:
        """Heuristic: return True when the query likely benefits from web context."""
        web_keywords = [
            "who is", "what is", "when did", "where is", "how to",
            "latest", "news", "weather", "price", "score", "define",
            "tell me about", "search", "look up", "find",
        ]
        lower = query.lower()
        return any(kw in lower for kw in web_keywords)
