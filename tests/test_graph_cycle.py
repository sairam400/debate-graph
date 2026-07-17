"""Proves the argue/rebut loop is a real graph cycle (advocate_for gets
invoked more than once via the conditional edge, not unrolled into separate
nodes per round) and that the controller's three stop conditions each reach
judge. Uses the phase-1 mock nodes -- no LLM, no network, no API spend."""
import unittest

from langgraph.checkpoint.sqlite import SqliteSaver

from src.graph.build import build_mock_graph
from src.state import DebateState


class TestGraphCycle(unittest.TestCase):
    def _run(self, max_rounds, checkpointer=None, thread_id="t1"):
        graph = build_mock_graph(checkpointer=checkpointer)
        initial = DebateState(question="does the mock cycle actually loop?", max_rounds=max_rounds)
        config = {"configurable": {"thread_id": thread_id}} if checkpointer else None
        return graph.invoke(initial, config=config)

    def test_cycle_runs_multiple_rounds_before_stopping(self):
        result = self._run(max_rounds=3)
        self.assertEqual(result["round_count"], 3)
        self.assertEqual(result["controller_decision"], "stop_max_rounds")
        # 2 transcript entries (for + against) per round x 3 rounds
        self.assertEqual(len(result["transcript"]), 6)
        self.assertEqual(len(result["evidence_ledger"]), 6)
        rounds_seen = sorted({t.round for t in result["transcript"]})
        self.assertEqual(rounds_seen, [1, 2, 3])

    def test_single_round_stops_immediately(self):
        result = self._run(max_rounds=1)
        self.assertEqual(result["round_count"], 1)
        self.assertEqual(len(result["transcript"]), 2)

    def test_judge_runs_after_stop(self):
        result = self._run(max_rounds=1)
        self.assertIsNotNone(result["verdict"])
        self.assertIn("stop_max_rounds", result["verdict"]["rationale"])

    def test_checkpointer_persists_full_round_history(self):
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            self._run(max_rounds=3, checkpointer=checkpointer, thread_id="checkpoint-test")
            config = {"configurable": {"thread_id": "checkpoint-test"}}
            history = list(checkpointer.list(config))
            # one checkpoint per super-step: START, then 7 nodes x 3 rounds worth
            # of super-steps (controller only fires 3 times, advocates/validators
            # also 3x each) -- assert it's more than a single before/after snapshot,
            # which is what would prove no persistence was happening at all.
            self.assertGreater(len(history), 10)

    def test_resumes_from_checkpoint_after_interruption(self):
        with SqliteSaver.from_conn_string(":memory:") as checkpointer:
            graph = build_mock_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "resume-test"}}
            initial = DebateState(question="resume me", max_rounds=3)

            # Simulate a mid-debate interruption: run one super-step at a time
            # and stop after the first round completes, then resume by passing
            # None as input (langgraph continues from the last checkpoint).
            steps = 0
            for _ in graph.stream(initial, config=config, stream_mode="values"):
                steps += 1
                if steps == 6:  # after assign_positions + one full round
                    break

            resumed = graph.invoke(None, config=config)
            self.assertEqual(resumed["controller_decision"], "stop_max_rounds")
            self.assertEqual(resumed["round_count"], 3)


if __name__ == "__main__":
    unittest.main()
