import json
import unittest

from langgraph.checkpoint.sqlite import SqliteSaver

from src.data.seed import DB_PATH, build as seed_db
from src.demo_cli import _print_debate, _save_trace, RUNS_DIR
from src.graph.build import build_llm_graph
from src.state import DebateState
from tests.test_llm_nodes import _PLAN


def setUpModule():
    if not DB_PATH.exists():
        seed_db()


class TestTraceSaving(unittest.TestCase):
    def test_trace_written_and_covers_full_run(self):
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = build_llm_graph(provider="mock", plan=_PLAN, checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "trace-test"}}
            initial = DebateState(question="Which category generated the most revenue?", max_rounds=1)
            result = graph.invoke(initial, config=config)

            _print_debate(result)  # just proving it doesn't crash on real result shapes

            path = _save_trace(graph, config, "trace-test-debate")
            self.assertTrue(path.exists())

            trace = json.loads(path.read_text(encoding="utf-8"))
            self.assertGreater(len(trace), 5)
            # The pre-START snapshot has no input applied yet, so it can't
            # validate against DebateState (question is required) -- that's
            # expected, not a bug; confirm it's exactly that one.
            self.assertIsNone(trace[0]["values"])
            self.assertEqual(trace[0]["next"], ["__start__"])
            self.assertIsNone(trace[1]["values"]["verdict"])
            self.assertEqual(trace[-1]["values"]["verdict"]["ruling"], "for")
            self.assertEqual(trace[-1]["next"], [])

        path.unlink()
        path.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
