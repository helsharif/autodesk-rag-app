import unittest

from src.agent import AutodeskRAGAgent


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


if __name__ == "__main__":
    unittest.main()
