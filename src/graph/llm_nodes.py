"""Real assign_positions/advocate/judge nodes, backed by any LangChain chat
model (ChatOllama, RateLimitedChatGroq, or MockChatModel -- all bindable via
.bind_tools the same way). Node factories close over the llm/side so the
graph wiring in build.py stays a plain dict of zero-arg-per-invoke callables.
"""
from ..config import SETTINGS
from ..providers.llm_json import JSONCompletionError, complete_json, extract_json
from ..state import DebateState, EvidenceEntry, TranscriptEntry
from . import prompts
from .tool_loop import describe_result, run_tool_loop


def make_assign_positions(llm):
    def assign_positions(state: DebateState) -> dict:
        user = (
            f"Question: {state.question}\n\n"
            "Frame two opposing positions a debate could resolve about this question."
        )
        text = run_tool_loop(
            llm, prompts.ASSIGN_POSITIONS_SYSTEM, user, max_calls=1,
            on_tool_result=lambda q, r: describe_result(r),
        )
        try:
            parsed = extract_json(text)
            positions = {"for": str(parsed["for"]), "against": str(parsed["against"])}
        except Exception:
            positions = {
                "for": f"An answer to: {state.question}",
                "against": f"A different answer to: {state.question}",
            }
        return {"positions": positions}

    return assign_positions


def _make_advocate(llm, side: str, max_calls: int):
    def advocate(state: DebateState) -> dict:
        round_in_progress = state.round_count + 1
        stage = "argument" if state.round_count == 0 else "rebuttal"
        new_evidence = []

        def on_tool_result(query, result):
            eid = f"E{len(state.evidence_ledger) + len(new_evidence) + 1}"
            description = describe_result(result)
            new_evidence.append(EvidenceEntry(
                id=eid, query=query, result=str(result["rows"]),
                interpretation=description, round=round_in_progress, side=side,
            ))
            return f"{eid}: {description}"

        opponent_side = "against" if side == "for" else "for"
        user = (
            f"Question: {state.question}\n"
            f"Your position ({side}): {state.positions.get(side, '')}\n"
            f"Opponent's position ({opponent_side}): {state.positions.get(opponent_side, '')}\n\n"
            f"Evidence ledger so far:\n{prompts.format_evidence_ledger(state.evidence_ledger)}\n\n"
            f"Debate so far:\n{prompts.format_transcript(state.transcript)}\n\n"
            f"Write your {stage} now."
        )
        raw_text = run_tool_loop(
            llm, prompts.ADVOCATE_SYSTEM, user, max_calls=max_calls, on_tool_result=on_tool_result
        )

        combined_ledger = state.evidence_ledger + new_evidence
        # Left unvalidated here on purpose: validate_for/validate_against are
        # separate graph nodes (see nodes.py, reused as-is) so the striker
        # runs as its own visible step in the trace, not hidden inside the
        # advocate's own turn.
        entry = TranscriptEntry(
            round=round_in_progress, side=side, stage=stage,
            raw_text=raw_text, validated_text=raw_text,
        )
        return {
            "evidence_ledger": combined_ledger,
            "transcript": state.transcript + [entry],
        }

    return advocate


def make_advocate_for(llm, max_calls: int = None):
    return _make_advocate(llm, "for", max_calls or SETTINGS.max_tool_calls_per_turn)


def make_advocate_against(llm, max_calls: int = None):
    return _make_advocate(llm, "against", max_calls or SETTINGS.max_tool_calls_per_turn)


def make_judge(llm):
    def judge(state: DebateState) -> dict:
        user = (
            f"Question: {state.question}\n"
            f"Position for: {state.positions.get('for', '')}\n"
            f"Position against: {state.positions.get('against', '')}\n\n"
            f"Validated transcript:\n{prompts.format_transcript(state.transcript)}\n\n"
            f"Evidence ledger:\n{prompts.format_evidence_ledger(state.evidence_ledger)}\n\n"
            f"Debate ended because: {state.controller_decision}. Render your verdict."
        )
        try:
            parsed = complete_json(llm, prompts.JUDGE_SYSTEM, user, retries=1)
            verdict = {
                "ruling": str(parsed["ruling"]),
                "confidence": float(parsed["confidence"]),
                "deciding_evidence": list(parsed.get("deciding_evidence", [])),
                "rationale": str(parsed["rationale"]),
                "final_answer": str(parsed["final_answer"]),
            }
        except (JSONCompletionError, KeyError, ValueError, TypeError) as exc:
            verdict = {
                "ruling": "unsettled",
                "confidence": 0.0,
                "deciding_evidence": [],
                "rationale": f"judge response could not be parsed as a verdict: {exc}",
                "final_answer": "cannot be determined",
            }
        return {"verdict": verdict}

    return judge
