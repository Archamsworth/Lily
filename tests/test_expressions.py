"""Tests for the expression/emotion parser (no external deps required)."""
import pytest
from app.avatar.expressions import (
    ExpressionEvent,
    TextToken,
    _map_to_vrm,
    parse_stream,
    split_response,
)


class TestMapToVrm:
    def test_smile_maps_to_happy(self):
        assert _map_to_vrm("smiles warmly") == "happy"

    def test_pout_maps_to_sad(self):
        assert _map_to_vrm("pouts") == "sad"

    def test_angry_maps_to_angry(self):
        assert _map_to_vrm("glares") == "angry"

    def test_surprise_maps_to_surprised(self):
        assert _map_to_vrm("gasps in surprise") == "surprised"

    def test_unknown_maps_to_neutral(self):
        assert _map_to_vrm("dances the macarena") == "neutral"

    def test_relax_maps_to_relaxed(self):
        assert _map_to_vrm("relaxes shoulders") == "relaxed"


class TestParseStream:
    def _stream(self, *tokens):
        def _gen():
            yield from tokens
        return list(parse_stream(_gen()))

    def test_plain_text_only(self):
        result = self._stream("Hello, world!")
        assert len(result) == 1
        assert isinstance(result[0], TextToken)
        assert result[0].text == "Hello, world!"

    def test_single_expression(self):
        result = self._stream("*smiles* Hi there!")
        assert any(isinstance(t, ExpressionEvent) and t.raw == "smiles" for t in result)
        assert any(isinstance(t, TextToken) and "Hi there!" in t.text for t in result)

    def test_expression_vrm_mapped(self):
        result = self._stream("*laughs loudly*")
        events = [t for t in result if isinstance(t, ExpressionEvent)]
        assert events[0].vrm_expression == "happy"

    def test_multiple_expressions(self):
        result = self._stream("*pouts* Where have you been? *folds arms*")
        events = [t for t in result if isinstance(t, ExpressionEvent)]
        assert len(events) == 2
        assert events[0].vrm_expression == "sad"
        assert events[1].vrm_expression == "angry"

    def test_split_token_across_boundary(self):
        """Expression marker split across two tokens must still be parsed."""
        result = self._stream("Hello *smi", "les* world")
        events = [t for t in result if isinstance(t, ExpressionEvent)]
        assert len(events) == 1
        assert events[0].raw == "smiles"

    def test_truly_empty_asterisks_not_parsed(self):
        """Consecutive asterisks with no content between them are not an expression."""
        result = self._stream("****")
        events = [t for t in result if isinstance(t, ExpressionEvent)]
        assert len(events) == 0


class TestSplitResponse:
    def test_full_split(self):
        text = "*smiles warmly* Sure! I'd be happy to help. *tilts head* What do you need?"
        spoken, expressions = split_response(text)
        assert "Sure! I'd be happy to help." in spoken
        assert "*smiles warmly*" not in spoken
        assert len(expressions) == 2
        assert expressions[0].vrm_expression == "happy"
        assert expressions[1].vrm_expression == "relaxed"

    def test_no_expressions(self):
        text = "Just a plain reply."
        spoken, expressions = split_response(text)
        assert spoken == text
        assert expressions == []
