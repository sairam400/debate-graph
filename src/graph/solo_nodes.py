"""The single-analyst baseline: analyst -> validate -> finalize. No
positions, no rounds, no controller -- but the same run_sql tool, evidence
ledger, and citation validator as the debate graph, so the 2x2 experiment
compares graph shape, not tooling.
"""
from ..state import EvidenceEntry, SoloState
from . import prompts
from .tool_loop import describe_result, run_tool_loop
from .validator import validate_argument


def make_analyst(llm, max_calls: int):
    def analyst(state: SoloState) -> dict:
        new_evidence = []

        def on_tool_result(query, result):
            eid = f"E{len(state.evidence_ledger) + len(new_evidence) + 1}"
            description = describe_result(result)
            new_evidence.append(EvidenceEntry(
                id=eid, query=query, result=str(result["rows"]),
                interpretation=description, round=1, side="solo",
            ))
            return f"{eid}: {description}"

        user = f"Question: {state.question}\n\nAnswer it now."
        raw_answer = run_tool_loop(
            llm, prompts.ANALYST_SYSTEM, user, max_calls=max_calls, on_tool_result=on_tool_result
        )

        return {
            "evidence_ledger": state.evidence_ledger + new_evidence,
            "raw_answer": raw_answer,
        }

    return analyst


def validate(state: SoloState) -> dict:
    result = validate_argument(state.raw_answer, state.evidence_ledger)
    return {
        "validated_answer": result["validated_text"],
        "citations": result["citations"],
        "struck_sentences": result["struck_sentences"],
    }


def finalize(state: SoloState) -> dict:
    answer = state.validated_answer.strip()
    if not answer:
        answer = "cannot be determined (no cited claim survived validation)"
    return {"final_answer": answer}
