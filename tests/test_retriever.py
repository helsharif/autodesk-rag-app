import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.retriever import RetrievedSource, search_documents


class RetrieverTests(unittest.TestCase):
    def test_search_documents_runs_dense_and_bm25_branches(self):
        dense_doc = Document(page_content="Dense result", metadata={"chunk_id": "dense"})
        bm25_doc = Document(page_content="BM25 result", metadata={"chunk_id": "bm25"})
        dense_source = RetrievedSource("Dense source", None, 0.9, "Dense result")
        bm25_source = RetrievedSource("BM25 source", None, 0.8, "BM25 result")

        with (
            patch("src.retriever._search_dense", return_value=[("dense", dense_doc, dense_source)]) as dense_mock,
            patch("src.retriever._search_bm25", return_value=[("bm25", bm25_doc, bm25_source)]) as bm25_mock,
        ):
            docs, sources = search_documents("AutoCAD Revit comparison", k=2)

        dense_mock.assert_called_once()
        bm25_mock.assert_called_once()
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(sources), 2)


if __name__ == "__main__":
    unittest.main()
