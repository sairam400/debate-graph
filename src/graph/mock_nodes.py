"""Phase-1 placeholder nodes. No LLM, no run_sql -- just enough deterministic
behavior to prove the graph's shape: assign_positions runs once, advocates
alternate, validators pass text through untouched (the real citation striker
is phase 2), and the argue/rebut pair repeats as a true cycle until the
controller (controller.py, already real) says stop.

Replaced node-by-node in later phases; the graph wiring in build.py does not
change when that happens.
"""
from ..state import DebateState, EvidenceEntry, TranscriptEntry


def assign_positions(state: DebateState) -> dict:
    return {
        "positions": {
            "for": f"(mock) affirms an answer to: {state.question}",
            "against": f"(mock) disputes the affirmed answer to: {state.question}",
        }
    }


def _mock_advocate(state: DebateState, side: str, stage: str) -> dict:
    round_in_progress = state.round_count + 1
    evidence_id = f"E{len(state.evidence_ledger) + 1}"
    evidence = EvidenceEntry(
        id=evidence_id,
        query=f"SELECT 'mock result for {side} round {round_in_progress}'",
        result="mock row",
        interpretation=f"(mock) supports the {side} position",
        round=round_in_progress,
        side=side,
    )
    text = f"(mock {stage} from {side}, round {round_in_progress}) [{evidence_id}] cites mock evidence."
    entry = TranscriptEntry(
        round=round_in_progress,
        side=side,
        stage=stage,
        raw_text=text,
        validated_text=text,
        citations=[evidence_id],
        struck_sentences=[],
    )
    return {
        "evidence_ledger": state.evidence_ledger + [evidence],
        "transcript": state.transcript + [entry],
    }


def advocate_for(state: DebateState) -> dict:
    stage = "argument" if state.round_count == 0 else "rebuttal"
    return _mock_advocate(state, "for", stage)


def advocate_against(state: DebateState) -> dict:
    stage = "argument" if state.round_count == 0 else "rebuttal"
    return _mock_advocate(state, "against", stage)


def validate_for(state: DebateState) -> dict:
    return {}


def validate_against(state: DebateState) -> dict:
    return {}


def judge(state: DebateState) -> dict:
    return {
        "verdict": {
            "ruling": "(mock) verdict pending real judge prompt",
            "confidence": 0.0,
            "deciding_evidence": [e.id for e in state.evidence_ledger],
            "rationale": f"(mock) stopped because {state.controller_decision}",
        }
    }
