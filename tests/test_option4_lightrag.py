import unittest
import asyncio
from unittest.mock import patch

from langchain_core.documents import Document

from src.agent import AutodeskRAGAgent
from src.config import LIGHTRAG_AUTODESK_WEB_MODE, LIGHTRAG_ONLY_MODE, OPTION_4_LABEL, OPTION_5_LABEL, SEARCH_MODE_OPTIONS
from src.lightrag_adapter import _run_async
from src.retriever import RetrievedSource


class Option4LightRAGTests(unittest.TestCase):
    def test_option4_is_exposed_as_search_mode(self):
        self.assertEqual(SEARCH_MODE_OPTIONS[OPTION_4_LABEL], LIGHTRAG_ONLY_MODE)

    def test_option5_is_lightrag_plus_autodesk_web(self):
        self.assertEqual(SEARCH_MODE_OPTIONS[OPTION_5_LABEL], LIGHTRAG_AUTODESK_WEB_MODE)

    def test_option4_uses_lightrag_retrieval_without_context_expansion(self):
        agent = AutodeskRAGAgent.__new__(AutodeskRAGAgent)
        agent.collection_name = "test"
        agent.search_mode = LIGHTRAG_ONLY_MODE
        doc = Document(page_content="AutoCAD is used for 2D drafting and 3D design.", metadata={})
        source = RetrievedSource("LightRAG mixed local corpus evidence", None, 1.0, doc.page_content)

        with (
            patch("src.agent.search_lightrag_mixed", return_value=([doc], [source])) as lightrag_mock,
            patch("src.agent.search_documents") as hybrid_mock,
        ):
            docs, sources, _ = agent._retrieve_local_documents("What is AutoCAD used for?")

        lightrag_mock.assert_called_once()
        hybrid_mock.assert_not_called()
        self.assertEqual(docs, [doc])
        self.assertEqual(sources, [source])

    def test_option4_route_never_requests_web(self):
        agent = AutodeskRAGAgent.__new__(AutodeskRAGAgent)
        agent.search_mode = LIGHTRAG_ONLY_MODE

        route = agent._route_query("What's the difference between AutoCAD and Maya?")

        self.assertTrue(route.needs_local)
        self.assertFalse(route.needs_web)
        self.assertFalse(route.abstain)
        self.assertIn("web search disabled", route.reason)

    def test_option5_uses_lightrag_retrieval_without_context_expansion(self):
        agent = AutodeskRAGAgent.__new__(AutodeskRAGAgent)
        agent.collection_name = "test"
        agent.search_mode = LIGHTRAG_AUTODESK_WEB_MODE
        doc = Document(page_content="AutoCAD is used for 2D drafting and 3D design.", metadata={})
        source = RetrievedSource("LightRAG mixed local corpus evidence", None, 1.0, doc.page_content)

        with (
            patch("src.agent.search_lightrag_mixed", return_value=([doc], [source])) as lightrag_mock,
            patch("src.agent.search_documents") as hybrid_mock,
        ):
            docs, sources, _ = agent._retrieve_local_documents("What is AutoCAD used for?")

        lightrag_mock.assert_called_once()
        hybrid_mock.assert_not_called()
        self.assertEqual(docs, [doc])
        self.assertEqual(sources, [source])

    def test_option5_web_query_is_scoped_to_autodesk(self):
        agent = AutodeskRAGAgent.__new__(AutodeskRAGAgent)
        agent.search_mode = LIGHTRAG_AUTODESK_WEB_MODE

        self.assertEqual(
            agent._web_query("What Autodesk products support BIM workflows?"),
            "site:autodesk.com Autodesk What Autodesk products support BIM workflows?",
        )

    def test_lightrag_async_runner_reuses_event_loop(self):
        async def loop_id():
            return id(asyncio.get_running_loop())

        self.assertEqual(_run_async(loop_id()), _run_async(loop_id()))


if __name__ == "__main__":
    unittest.main()
