"""Decides continue/stop after each round. Deterministic and auditable on
purpose: no extra LLM call judges the debate's own progress, so the stop
reason is always traceable to a rule, not a model's mood.

Precedence when multiple conditions fire in the same round: max_rounds beats
concession beats no_new_evidence, since a round that both hits the cap and
sees a concession should be reported as capped (the more specific, more
surprising signal) -- concession is checked before no_new_evidence for the
same reason: a concession that happens not to cite new evidence is still a
concession, not a stall.
"""
from ..state import DebateState, TranscriptEntry

_CONCESSION_MARKERS = (
    "i concede",
    "the evidence supports the other position",
    "i withdraw this position",
)


def _conceded(entries: list[TranscriptEntry]) -> bool:
    return any(
        marker in entry.validated_text.lower()
        for entry in entries
        for marker in _CONCESSION_MARKERS
    )


def controller(state: DebateState) -> dict:
    new_round = state.round_count + 1
    round_transcript = [t for t in state.transcript if t.round == new_round]
    round_evidence = [e for e in state.evidence_ledger if e.round == new_round]

    if new_round >= state.max_rounds:
        decision = "stop_max_rounds"
    elif _conceded(round_transcript):
        decision = "stop_concession"
    elif not round_evidence:
        decision = "stop_no_new_evidence"
    else:
        decision = "continue"

    return {"round_count": new_round, "controller_decision": decision}


def route_after_controller(state: DebateState) -> str:
    return "continue" if state.controller_decision == "continue" else "stop"
