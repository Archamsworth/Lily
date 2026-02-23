"""Tests for the RAG retriever (uses mocks – no network required)."""
from unittest.mock import MagicMock, patch

import pytest

from app.llm.rag import RAGRetriever


class TestShouldSearch:
    def test_news_query_triggers_search(self):
        assert RAGRetriever.should_search("What is the latest news about AI?")

    def test_how_to_triggers_search(self):
        assert RAGRetriever.should_search("How to bake bread")

    def test_casual_greeting_does_not_trigger(self):
        assert not RAGRetriever.should_search("Hey, how are you?")

    def test_tell_me_about_triggers(self):
        assert RAGRetriever.should_search("Tell me about the Eiffel Tower")


class TestRAGRetriever:
    def test_disabled_returns_empty(self):
        retriever = RAGRetriever({"enabled": False})
        assert retriever.get_context("anything") == ""

    @patch("app.llm.rag.DDGS")
    @patch("app.llm.rag.requests.get")
    def test_get_context_returns_text(self, mock_get, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = lambda s: s
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = [
            {"title": "Test", "href": "https://example.com", "body": "Fallback text"},
        ]
        mock_ddgs_cls.return_value = mock_ddgs

        mock_response = MagicMock()
        mock_response.text = "<html><body><p>Some page content here.</p></body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        retriever = RAGRetriever({"enabled": True, "search_results": 1})
        context = retriever.get_context("What is Python?")
        assert context != ""
        # Verify the mock source URL is cited in the returned context
        assert context.find("https://example.com") != -1

    @patch("app.llm.rag.DDGS")
    def test_empty_search_results_returns_empty(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = lambda s: s
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs

        retriever = RAGRetriever({"enabled": True})
        assert retriever.get_context("something") == ""
