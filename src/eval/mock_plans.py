"""Scripted plans for `python -m src.eval.run_eval --mock` -- a wiring check
for the harness itself (results file, resume-skip, scoring, summary table),
not a model comparison. MockChatModel doesn't read the prompt, so the same
plan replays identically for every one of the 14 cases; only the *shape*
(how many LLM calls debate/solo make) has to match reality.

Debate: assign_positions (1, no tool call) + 3 rounds x 2 sides x (1 tool
call + 1 final) = 1 + 12 = 13 calls, then judge (1) = 14. Assumes the
default max_rounds=3 and that the mock advocates always add evidence and
never concede, so the controller always runs the full 3 rounds.
"""
import json

_ADVOCATE_TURN = lambda eid: [
    {"tool_calls": [{"name": "run_sql_tool", "args": {"query": "SELECT 1"}}]},
    {"content": f"[{eid}] Mock evidence supports this turn's position."},
]

DEBATE_MOCK_PLAN = (
    [{"content": json.dumps({"for": "mock claim", "against": "mock counter-claim"})}]
    + _ADVOCATE_TURN("E1") + _ADVOCATE_TURN("E2")
    + _ADVOCATE_TURN("E3") + _ADVOCATE_TURN("E4")
    + _ADVOCATE_TURN("E5") + _ADVOCATE_TURN("E6")
    + [{"content": json.dumps({
        "ruling": "unsettled",
        "confidence": 0.5,
        "deciding_evidence": ["E1"],
        "rationale": "mock wiring check, not a real verdict",
        "final_answer": "cannot be determined (mock run)",
    })}]
)

SOLO_MOCK_PLAN = [
    {"tool_calls": [{"name": "run_sql_tool", "args": {"query": "SELECT 1"}}]},
    {"content": "[E1] Mock evidence supports this answer."},
]

MOCK_MODELS = {
    "mock": ("mock", {"solo": SOLO_MOCK_PLAN, "debate": DEBATE_MOCK_PLAN}),
}
