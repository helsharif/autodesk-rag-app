import unittest

from langchain_core.documents import Document

from src.agent import AutodeskRAGAgent
from src.retriever import RetrievedSource


class CompareRouterTests(unittest.TestCase):
    def test_difference_between_products_generates_balanced_subqueries(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("What is the difference between AutoCAD and Revit?")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["AutoCAD", "Revit"])
        self.assertLessEqual(len(plan.subqueries), 4)
        self.assertTrue(any("AutoCAD" in query and "Revit" in query for query in plan.subqueries))

    def test_vs_products_generalizes_to_another_pair(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("Maya vs 3ds Max for animation workflows")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["Maya", "3ds Max"])
        self.assertLessEqual(len(plan.subqueries), 4)
        self.assertTrue(any("Maya" in query for query in plan.subqueries))
        self.assertTrue(any("3ds Max" in query for query in plan.subqueries))

    def test_selection_query_preserves_numeric_product_name(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("Which is better, Fusion 360 or Inventor?")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["Fusion 360", "Inventor"])

    def test_non_compare_query_does_not_trigger(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("What is Autodesk Navisworks used for?")

        self.assertFalse(plan.is_compare)
        self.assertEqual(plan.products, [])
        self.assertEqual(plan.subqueries, [])

    def test_compare_entity_support_accepts_autodesk_prefixes(self):
        self.assertTrue(
            AutodeskRAGAgent._compare_entities_supported(
                ["AutoCAD", "Maya"],
                ["Autodesk AutoCAD", "Autodesk Maya"],
            )
        )

    def test_compare_entity_support_accepts_entity_objects(self):
        self.assertTrue(
            AutodeskRAGAgent._compare_entities_supported(
                ["AutoCAD", "Maya"],
                [{"entity": "Autodesk AutoCAD"}, {"name": "Autodesk Maya"}],
            )
        )

    def test_compare_entity_support_requires_both_products(self):
        self.assertFalse(
            AutodeskRAGAgent._compare_entities_supported(
                ["AutoCAD", "Maya"],
                ["Autodesk AutoCAD"],
            )
        )

    def test_post_rerank_balance_reinserts_missing_product(self):
        autocad_doc = Document(page_content="AutoCAD creates precise 2D drawings and 3D models.", metadata={"chunk_id": "a"})
        autocad_doc_2 = Document(page_content="AutoCAD includes drafting and documentation workflows.", metadata={"chunk_id": "b"})
        maya_doc = Document(page_content="Maya provides 3D animation, modeling, simulation, and rendering tools.", metadata={"chunk_id": "c"})
        sources = [
            RetrievedSource("AutoCAD source 1", None, 0.9, "AutoCAD creates precise 2D drawings and 3D models."),
            RetrievedSource("AutoCAD source 2", None, 0.8, "AutoCAD includes drafting and documentation workflows."),
            RetrievedSource("Maya source", None, 0.7, "Maya provides 3D animation, modeling, simulation, and rendering tools."),
        ]

        docs, balanced_sources = AutodeskRAGAgent._ensure_compare_balance_after_rerank(
            [autocad_doc, autocad_doc_2],
            sources[:2],
            [autocad_doc, autocad_doc_2, maya_doc],
            sources,
            ["AutoCAD", "Maya"],
            limit=2,
        )

        combined_text = " ".join(doc.page_content for doc in docs)
        self.assertIn("AutoCAD", combined_text)
        self.assertIn("Maya", combined_text)
        self.assertEqual(len(balanced_sources), 2)

    def test_compare_evidence_coverage_accepts_separate_product_chunks(self):
        docs = [
            Document(page_content="AutoCAD creates precise 2D drawings and 3D models for design and documentation workflows.", metadata={"title": "AutoCAD overview"}),
            Document(page_content="Revit supports BIM workflows for architecture and engineering projects with modeling and documentation tools.", metadata={"title": "Revit overview"}),
        ]
        sources = [
            RetrievedSource("AutoCAD source", None, 0.9, docs[0].page_content),
            RetrievedSource("Revit source", None, 0.8, docs[1].page_content),
        ]

        self.assertTrue(
            AutodeskRAGAgent._compare_evidence_has_entity_coverage(
                docs,
                sources,
                ["AutoCAD", "Revit"],
            )
        )

    def test_compare_evidence_coverage_requires_both_products(self):
        docs = [
            Document(page_content="AutoCAD creates precise 2D drawings and 3D models for design and documentation workflows.", metadata={"title": "AutoCAD overview"}),
        ]
        sources = [RetrievedSource("AutoCAD source", None, 0.9, docs[0].page_content)]

        self.assertFalse(
            AutodeskRAGAgent._compare_evidence_has_entity_coverage(
                docs,
                sources,
                ["AutoCAD", "Revit"],
            )
        )


if __name__ == "__main__":
    unittest.main()
