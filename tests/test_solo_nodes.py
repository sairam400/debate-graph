import unittest

from src.data.seed import DB_PATH, build as seed_db
from src.graph.solo_build import build_solo_graph
from src.state import SoloState


def setUpModule():
    if not DB_PATH.exists():
        seed_db()


_PLAN = [
    {"tool_calls": [{"name": "run_sql_tool", "args": {
        "query": "SELECT category, ROUND(SUM(oi.quantity*oi.unit_price),2) AS rev "
                 "FROM order_items oi JOIN products p ON p.id=oi.product_id "
                 "GROUP BY category ORDER BY rev DESC LIMIT 1",
    }}]},
    {"content": "[E1] Electronics generated the most revenue of any category. "
                "This is a well-known fact around the office."},
]


class TestSoloNodes(unittest.TestCase):
    def test_full_solo_run_via_mock_provider(self):
        graph = build_solo_graph(provider="mock", plan=_PLAN)
        initial = SoloState(question="Which product category generated the most revenue?")
        result = graph.invoke(initial)

        self.assertEqual(len(result["evidence_ledger"]), 1)
        self.assertEqual(result["evidence_ledger"][0].id, "E1")
        self.assertTrue(result["evidence_ledger"][0].query.upper().startswith("SELECT"))

        self.assertIn("[E1]", result["validated_answer"])
        self.assertNotIn("well-known fact", result["validated_answer"])
        self.assertEqual(len(result["struck_sentences"]), 1)

        self.assertEqual(result["final_answer"], result["validated_answer"])

    def test_finalize_falls_back_when_everything_struck(self):
        plan = [{"content": "This has no citation at all."}]
        graph = build_solo_graph(provider="mock", plan=plan)
        initial = SoloState(question="Some question")
        result = graph.invoke(initial)

        self.assertEqual(result["validated_answer"], "")
        self.assertIn("cannot be determined", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
