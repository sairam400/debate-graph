"""assign_positions and judge are still templated placeholders -- real
prompts land in phase 3. advocate_for/against and validate_for/against are
real as of phase 2: advocates run actual run_sql queries against the seeded
db and populate the evidence ledger for real, and the validators run the
real citation striker (validator.py). The queries advocates run here are a
small fixed rotation, not yet chosen by a model -- that's the one piece
phase 3 replaces, and it's why every argument still includes one sentence
with no citation: proving the validator actually strikes something, not
just passes text through.
"""
from ..state import DebateState, EvidenceEntry, TranscriptEntry
from ..tools.sql import run_sql
from .validator import validate_argument

_FOR_QUERIES = [
    "SELECT category, ROUND(SUM(oi.quantity*oi.unit_price),2) AS rev "
    "FROM order_items oi JOIN products p ON p.id=oi.product_id "
    "GROUP BY category ORDER BY rev DESC LIMIT 1",
    "SELECT COUNT(*) AS n FROM orders",
    "SELECT ROUND(AVG(unit_price),2) AS avg_price FROM products",
]
_AGAINST_QUERIES = [
    "SELECT reason, COUNT(*) AS n FROM returns GROUP BY reason ORDER BY n DESC LIMIT 1",
    "SELECT COUNT(*) AS n FROM returns",
    "SELECT category, COUNT(*) AS n FROM products GROUP BY category ORDER BY n DESC LIMIT 1",
]


def assign_positions(state: DebateState) -> dict:
    return {
        "positions": {
            "for": f"(template) affirms an answer to: {state.question}",
            "against": f"(template) disputes the affirmed answer to: {state.question}",
        }
    }


def _describe_result(result: dict) -> str:
    if not result["rows"]:
        return "the query returned no rows"
    first = result["rows"][0]
    return ", ".join(f"{col}={val}" for col, val in zip(result["columns"], first))


def _advocate(state: DebateState, side: str) -> dict:
    round_in_progress = state.round_count + 1
    stage = "argument" if state.round_count == 0 else "rebuttal"
    queries = _FOR_QUERIES if side == "for" else _AGAINST_QUERIES
    query = queries[(round_in_progress - 1) % len(queries)]

    result = run_sql(query)
    description = _describe_result(result)

    evidence_id = f"E{len(state.evidence_ledger) + 1}"
    evidence = EvidenceEntry(
        id=evidence_id,
        query=query,
        result=str(result["rows"]),
        interpretation=description,
        round=round_in_progress,
        side=side,
    )

    raw_text = (
        f"[{evidence_id}] The data shows {description}. "
        f"This clearly favors the {side} position, as anyone can see."
    )
    entry = TranscriptEntry(
        round=round_in_progress,
        side=side,
        stage=stage,
        raw_text=raw_text,
        validated_text=raw_text,
        citations=[],
        struck_sentences=[],
    )
    return {
        "evidence_ledger": state.evidence_ledger + [evidence],
        "transcript": state.transcript + [entry],
    }


def advocate_for(state: DebateState) -> dict:
    return _advocate(state, "for")


def advocate_against(state: DebateState) -> dict:
    return _advocate(state, "against")


def _validate_side(state: DebateState, side: str) -> dict:
    round_in_progress = state.round_count + 1
    idx = max(
        i
        for i, t in enumerate(state.transcript)
        if t.round == round_in_progress and t.side == side
    )
    entry = state.transcript[idx]
    result = validate_argument(entry.raw_text, state.evidence_ledger)
    updated = entry.model_copy(update=result)
    new_transcript = list(state.transcript)
    new_transcript[idx] = updated
    return {"transcript": new_transcript}


def validate_for(state: DebateState) -> dict:
    return _validate_side(state, "for")


def validate_against(state: DebateState) -> dict:
    return _validate_side(state, "against")


def judge(state: DebateState) -> dict:
    return {
        "verdict": {
            "ruling": "(template) verdict pending real judge prompt",
            "confidence": 0.0,
            "deciding_evidence": [e.id for e in state.evidence_ledger],
            "rationale": f"(template) stopped because {state.controller_decision}",
        }
    }
