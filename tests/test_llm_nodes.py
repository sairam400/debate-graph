"""Full debate through build_llm_graph with provider='mock': proves the real
prompts/parsing/tool-loop wiring works without needing Ollama or Groq. Real
tool calls still execute against the real seeded db; only the LLM's
decisions are scripted."""
import json
import unittest

from src.data.seed import DB_PATH, build as seed_db
from src.graph.build import build_llm_graph
from src.state import DebateState


def setUpModule():
    if not DB_PATH.exists():
        seed_db()


_PLAN = [
    # assign_positions: no exploratory query, straight to JSON.
    {"content": json.dumps({
        "for": "Electronics generated the most revenue of any category.",
        "against": "A different category generated the most revenue.",
    })},
    # advocate_for, round 1 argument: one tool call, then final text.
    {"tool_calls": [{"name": "run_sql_tool", "args": {
        "query": "SELECT category, ROUND(SUM(oi.quantity*oi.unit_price),2) AS rev "
                 "FROM order_items oi JOIN products p ON p.id=oi.product_id "
                 "GROUP BY category ORDER BY rev DESC LIMIT 1",
    }}]},
    {"content": "[E1] The top category by revenue supports the for position. "
                "This is a strong point in favor of the claim."},
    # advocate_against, round 1 argument.
    {"tool_calls": [{"name": "run_sql_tool", "args": {"query": "SELECT COUNT(*) AS n FROM returns"}}]},
    {"content": "[E2] Return volume is a separate consideration entirely. "
                "This does not really settle the revenue question though."},
    # judge: final JSON verdict.
    {"content": json.dumps({
        "ruling": "for",
        "confidence": 0.7,
        "deciding_evidence": ["E1"],
        "rationale": "E1 directly answers the question; E2 is unrelated to revenue.",
        "final_answer": "Electronics",
    })},
]


class TestLlmNodesMock(unittest.TestCase):
    def test_full_debate_via_mock_provider(self):
        graph = build_llm_graph(provider="mock", plan=_PLAN)
        initial = DebateState(question="Which category generated the most revenue?", max_rounds=1)
        result = graph.invoke(initial)

        self.assertEqual(
            result["positions"]["for"],
            "Electronics generated the most revenue of any category.",
        )

        self.assertEqual(len(result["transcript"]), 2)
        for_entry = next(t for t in result["transcript"] if t.side == "for")
        against_entry = next(t for t in result["transcript"] if t.side == "against")

        self.assertIn("[E1]", for_entry.validated_text)
        self.assertNotIn("strong point in favor", for_entry.validated_text)
        self.assertEqual(len(for_entry.struck_sentences), 1)

        self.assertIn("[E2]", against_entry.validated_text)
        self.assertEqual(len(against_entry.struck_sentences), 1)

        self.assertEqual(len(result["evidence_ledger"]), 2)
        self.assertEqual(result["evidence_ledger"][0].id, "E1")
        self.assertEqual(result["evidence_ledger"][1].id, "E2")
        self.assertTrue(result["evidence_ledger"][0].query.upper().startswith("SELECT"))

        self.assertEqual(result["verdict"]["ruling"], "for")
        self.assertEqual(result["verdict"]["deciding_evidence"], ["E1"])
        self.assertEqual(result["verdict"]["final_answer"], "Electronics")
        self.assertEqual(result["controller_decision"], "stop_max_rounds")


if __name__ == "__main__":
    unittest.main()
