import unittest

from src.graph.controller import controller, route_after_controller
from src.state import DebateState, EvidenceEntry, TranscriptEntry


def _entry(round_, side, text, stage="argument"):
    return TranscriptEntry(
        round=round_, side=side, stage=stage, raw_text=text, validated_text=text
    )


def _evidence(round_, side):
    return EvidenceEntry(
        id=f"E{round_}{side}", query="SELECT 1", result="1", interpretation="x",
        round=round_, side=side,
    )


class TestController(unittest.TestCase):
    def test_continue_when_evidence_grew_and_no_concession(self):
        state = DebateState(
            question="q",
            max_rounds=5,
            round_count=0,
            transcript=[_entry(1, "for", "[E1] argues something."),
                        _entry(1, "against", "[E2] rebuts it.")],
            evidence_ledger=[_evidence(1, "for"), _evidence(1, "against")],
        )
        result = controller(state)
        self.assertEqual(result["controller_decision"], "continue")
        self.assertEqual(result["round_count"], 1)
        self.assertEqual(route_after_controller(state.model_copy(update=result)), "continue")

    def test_stop_max_rounds_takes_priority(self):
        state = DebateState(
            question="q",
            max_rounds=1,
            round_count=0,
            transcript=[_entry(1, "for", "[E1] argues, i concede nothing though.")],
            evidence_ledger=[_evidence(1, "for")],
        )
        result = controller(state)
        self.assertEqual(result["controller_decision"], "stop_max_rounds")
        self.assertEqual(route_after_controller(state.model_copy(update=result)), "stop")

    def test_stop_concession(self):
        state = DebateState(
            question="q",
            max_rounds=5,
            round_count=0,
            transcript=[_entry(1, "for", "[E1] argues something."),
                        _entry(1, "against", "[E2] I concede this point.")],
            evidence_ledger=[_evidence(1, "for"), _evidence(1, "against")],
        )
        result = controller(state)
        self.assertEqual(result["controller_decision"], "stop_concession")

    def test_stop_no_new_evidence(self):
        state = DebateState(
            question="q",
            max_rounds=5,
            round_count=0,
            transcript=[_entry(1, "for", "restates a prior point with no new query."),
                        _entry(1, "against", "restates its rebuttal, nothing new.")],
            evidence_ledger=[],
        )
        result = controller(state)
        self.assertEqual(result["controller_decision"], "stop_no_new_evidence")

    def test_only_current_round_evidence_counts(self):
        # Evidence exists, but all from a prior round -- this round added none.
        state = DebateState(
            question="q",
            max_rounds=5,
            round_count=1,
            transcript=[_entry(2, "for", "no citation needed, just restating."),
                        _entry(2, "against", "same, nothing new.")],
            evidence_ledger=[_evidence(1, "for")],
        )
        result = controller(state)
        self.assertEqual(result["controller_decision"], "stop_no_new_evidence")


if __name__ == "__main__":
    unittest.main()
