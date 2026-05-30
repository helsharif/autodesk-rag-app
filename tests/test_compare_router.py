import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from src.agent import AutodeskRAGAgent
from src.retriever import RetrievedSource


class CompareRouterTests(unittest.TestCase):
    def test_difference_between_products_generates_balanced_subqueries(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("What is the difference between AutoCAD and Revit?")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["AutoCAD", "Revit"])
        self.assertLessEqual(len(plan.subqueries), 5)
        self.assertIn("What is AutoCAD and its main features?", plan.subqueries)
        self.assertIn("What is Revit and its main features?", plan.subqueries)
        self.assertTrue(any("AutoCAD" in query and "Revit" in query for query in plan.subqueries))

    def test_generic_products_generate_standalone_feature_queries(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("What is the difference between Product A and Product B?")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["Product A", "Product B"])
        self.assertIn("What is Product A and its main features?", plan.subqueries)
        self.assertIn("What is Product B and its main features?", plan.subqueries)
        self.assertTrue(any("Product A" in query and "Product B" in query for query in plan.subqueries))

    def test_vs_products_generalizes_to_another_pair(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("Maya vs 3ds Max for animation workflows")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["Maya", "3ds Max"])
        self.assertLessEqual(len(plan.subqueries), 5)
        self.assertIn("What is Maya and its main features?", plan.subqueries)
        self.assertIn("What is 3ds Max and its main features?", plan.subqueries)
        self.assertTrue(any("Maya" in query and "3ds Max" in query for query in plan.subqueries))

    def test_compare_for_domain_generates_workflow_feature_queries(self):
        plan = AutodeskRAGAgent._compare_retrieval_plan("Compare Fusion and Inventor for mechanical design workflows")

        self.assertTrue(plan.is_compare)
        self.assertEqual(plan.products[:2], ["Fusion", "Inventor"])
        self.assertIn("What is Fusion and its main features?", plan.subqueries)
        self.assertIn("What is Inventor and its main features?", plan.subqueries)
        self.assertIn("Fusion mechanical design workflows features", plan.subqueries)
        self.assertIn("Inventor mechanical design workflows features", plan.subqueries)
        self.assertIn("Fusion Inventor comparison mechanical design workflows", plan.subqueries)

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

    def test_compare_retrieval_searches_each_focused_query(self):
        agent = AutodeskRAGAgent.__new__(AutodeskRAGAgent)
        agent.collection_name = "test"
        docs = [
            Document(page_content="AutoCAD creates precise 2D drawings and 3D models for design and documentation workflows.", metadata={"title": "AutoCAD overview", "chunk_id": "a"}),
            Document(page_content="Revit supports BIM workflows for architecture and engineering projects with modeling and documentation tools.", metadata={"title": "Revit overview", "chunk_id": "b"}),
        ]
        sources = [
            RetrievedSource("AutoCAD source", None, 0.9, docs[0].page_content),
            RetrievedSource("Revit source", None, 0.8, docs[1].page_content),
        ]

        with patch("src.agent.search_documents", return_value=(docs, sources)) as search_mock:
            retrieved_docs, retrieved_sources, plan = agent._retrieve_local_documents("What's the difference between AutoCAD and Revit?")

        self.assertTrue(plan.is_compare)
        self.assertEqual(search_mock.call_count, 1 + len(plan.subqueries))
        searched_queries = [call.args[0] for call in search_mock.call_args_list]
        self.assertIn("What's the difference between AutoCAD and Revit?", searched_queries)
        self.assertIn("What is AutoCAD and its main features?", searched_queries)
        self.assertIn("What is Revit and its main features?", searched_queries)
        self.assertEqual(len(retrieved_docs), 2)
        self.assertEqual(len(retrieved_sources), 2)

    def test_compare_retrieval_tags_product_feature_subquery_results(self):
        agent = AutodeskRAGAgent.__new__(AutodeskRAGAgent)
        agent.collection_name = "test"
        doc = Document(page_content="AutoCAD creates precise 2D drawings and 3D models for design and documentation workflows.", metadata={"title": "AutoCAD overview", "chunk_id": "a"})
        source = RetrievedSource("AutoCAD source", None, 0.9, doc.page_content)

        with patch("src.agent.search_documents", return_value=([doc], [source])):
            retrieved_docs, _, plan = agent._retrieve_local_documents("What's the difference between AutoCAD and Revit?")

        target_products = {doc.metadata.get("compare_target_product") for doc in retrieved_docs if doc.metadata.get("compare_target_product")}
        self.assertTrue(plan.is_compare)
        self.assertIn("AutoCAD", target_products)

    def test_post_rerank_balance_prefers_targeted_product_evidence(self):
        revit_doc = Document(
            page_content="Bring AutoCAD files in or out of Revit for project stakeholders.",
            metadata={"chunk_id": "r", "compare_target_product": "Revit"},
        )
        incidental_autocad_doc = Document(
            page_content="AutoCAD vs Fusion 360 includes reusable block libraries and other modeling software capabilities.",
            metadata={"chunk_id": "af"},
        )
        targeted_autocad_doc = Document(
            page_content="AutoCAD creates precise 2D drawings and 3D models for design and documentation workflows.",
            metadata={"chunk_id": "a", "compare_target_product": "AutoCAD"},
        )
        sources = [
            RetrievedSource("Revit source", None, 0.9, revit_doc.page_content),
            RetrievedSource("AutoCAD vs Fusion source", None, 0.8, incidental_autocad_doc.page_content),
            RetrievedSource("AutoCAD source", None, 0.7, targeted_autocad_doc.page_content),
        ]

        docs, _ = AutodeskRAGAgent._ensure_compare_balance_after_rerank(
            [revit_doc, incidental_autocad_doc],
            sources[:2],
            [revit_doc, incidental_autocad_doc, targeted_autocad_doc],
            sources,
            ["AutoCAD", "Revit"],
            limit=2,
        )

        target_products = {doc.metadata.get("compare_target_product") for doc in docs}
        self.assertIn("AutoCAD", target_products)
        self.assertIn("Revit", target_products)

    def test_targeted_feature_evidence_generalizes_to_other_product_pairs(self):
        fusion_doc = Document(
            page_content="Fusion provides integrated CAD, CAM, CAE, and PCB tools for product development workflows.",
            metadata={"chunk_id": "f", "compare_target_product": "Fusion"},
        )
        inventor_doc = Document(
            page_content="Inventor provides mechanical design, 3D CAD, simulation, and documentation capabilities.",
            metadata={"chunk_id": "i", "compare_target_product": "Inventor"},
        )
        incidental_doc = Document(
            page_content="Fusion appears in a broad comparison table with other products.",
            metadata={"chunk_id": "x"},
        )
        sources = [
            RetrievedSource("Fusion incidental source", None, 0.95, incidental_doc.page_content),
            RetrievedSource("Inventor source", None, 0.9, inventor_doc.page_content),
            RetrievedSource("Fusion source", None, 0.8, fusion_doc.page_content),
        ]

        docs, _ = AutodeskRAGAgent._ensure_compare_balance_after_rerank(
            [incidental_doc, inventor_doc],
            sources[:2],
            [incidental_doc, inventor_doc, fusion_doc],
            sources,
            ["Fusion", "Inventor"],
            limit=2,
        )

        target_products = {doc.metadata.get("compare_target_product") for doc in docs}
        self.assertIn("Fusion", target_products)
        self.assertIn("Inventor", target_products)


if __name__ == "__main__":
    unittest.main()
