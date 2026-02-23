"""Expression / emotion streaming parser.

The LLM may embed *expression* markers inside its response text, e.g.:

    *smiles warmly* Sure, let me help! *tilts head* What would you like to know?

This module:
1. Parses those markers out of the raw token stream.
2. Yields structured ``ExpressionEvent`` objects that the frontend uses to
   animate the VRM avatar.
3. Yields the cleaned spoken text (without markers) for TTS.

Supported VRM expression names (mapped from natural language):
    happy, sad, angry, surprised, relaxed, neutral, blink, blinkLeft,
    blinkRight, aa, ih, ou, ee, oh  (vowel mouth shapes for lip-sync)

Any unrecognised emotion is sent as-is to the frontend for best-effort
animation; the frontend falls back to "neutral" when it cannot map the name.
"""
from __future__ import annotations

import re
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Union


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TextToken:
    """A plain text token to be spoken / displayed."""
    text: str


@dataclass
class ExpressionEvent:
    """An expression or emotion extracted from a *…* marker."""
    raw: str                    # Original text inside the asterisks
    vrm_expression: str         # Mapped VRM blend-shape name
    description: str = ""       # Human-readable description for the animation AI


ParsedToken = Union[TextToken, ExpressionEvent]


# ---------------------------------------------------------------------------
# Keyword → VRM blend-shape mapping
# ---------------------------------------------------------------------------

_EXPRESSION_MAP: dict[str, str] = {
    # Positive
    "smile": "happy", "smil": "happy", "grin": "happy", "laugh": "happy",
    "giggle": "happy", "chuckle": "happy", "beam": "happy", "joy": "happy",
    "happy": "happy", "pleased": "happy", "excited": "happy",
    # Sad
    "sad": "sad", "cry": "sad", "tear": "sad", "sigh": "sad",
    "pout": "sad", "frown": "sad", "sulk": "sad", "downcast": "sad",
    # Angry
    "angry": "angry", "anger": "angry", "cross": "angry", "furious": "angry",
    "glare": "angry", "scowl": "angry", "huff": "angry",
    # Surprised
    "surprise": "surprised", "surprised": "surprised", "shock": "surprised",
    "gasp": "surprised", "wide-eyed": "surprised", "wow": "surprised",
    # Relaxed / neutral
    "relax": "relaxed", "calm": "relaxed", "peaceful": "relaxed",
    "neutral": "neutral", "blank": "neutral",
    # Physical gestures mapped to nearest VRM expression
    "tilt": "relaxed", "nod": "happy", "wink": "happy",
    "fold": "angry",   # e.g. "folds arms"
    "cross arm": "angry", "arms crossed": "angry",
}


def _map_to_vrm(raw: str) -> str:
    """Map a raw expression string to a VRM blend-shape name."""
    lower = raw.lower()
    for keyword, vrm_name in _EXPRESSION_MAP.items():
        if keyword in lower:
            return vrm_name
    return "neutral"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_MARKER_RE = re.compile(r"(\*[^*\n]+\*)")


def parse_stream(token_stream: Generator[str, None, None]) -> Generator[ParsedToken, None, None]:
    """Parse a raw LLM token stream, yielding ``TextToken`` and ``ExpressionEvent``.

    Because tokens may split mid-marker we buffer across token boundaries.
    """
    buffer = ""
    for token in token_stream:
        buffer += token
        # Flush complete markers and text fragments
        while True:
            match = _MARKER_RE.search(buffer)
            if not match:
                # Yield text before any potential incomplete marker
                safe_end = _safe_flush_length(buffer)
                if safe_end > 0:
                    yield TextToken(buffer[:safe_end])
                    buffer = buffer[safe_end:]
                break
            # Yield text before the marker
            if match.start() > 0:
                yield TextToken(buffer[: match.start()])
            # Yield the expression event
            raw = match.group(1)[1:-1]  # Strip surrounding *
            yield ExpressionEvent(
                raw=raw,
                vrm_expression=_map_to_vrm(raw),
                description=raw,
            )
            buffer = buffer[match.end():]

    # Flush remaining buffer
    if buffer:
        yield TextToken(buffer)


def _safe_flush_length(text: str) -> int:
    """Return how many chars from the start of *text* are safe to emit.

    We hold back content after the last ``*`` in case the marker is split
    across future tokens.
    """
    star_pos = text.rfind("*")
    return star_pos if star_pos >= 0 else len(text)


def split_response(full_text: str) -> tuple[str, list[ExpressionEvent]]:
    """Split a complete LLM response into spoken text and a list of expressions."""

    def _single_token():
        yield full_text

    spoken_parts: list[str] = []
    expressions: list[ExpressionEvent] = []

    for item in parse_stream(_single_token()):
        if isinstance(item, TextToken):
            spoken_parts.append(item.text)
        elif isinstance(item, ExpressionEvent):
            expressions.append(item)

    return " ".join(spoken_parts).strip(), expressions
