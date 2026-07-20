"""Run one debate end to end and print the transcript + verdict. Also writes
the full graph execution trace (every state snapshot langgraph's checkpoint
history already tracks) to runs/<debate_id>/trace.json -- reusing the
checkpointer as the trace source instead of hand-rolling separate bookkeeping.

Examples:
  python -m src.demo_cli --provider ollama --question "Which category generated the most revenue?"
  python -m src.demo_cli --provider groq --question "..." --max-rounds 4
"""
import argparse
import json
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from .config import SETTINGS
from .data.seed import DB_PATH as SEED_DB_PATH, build as seed_db
from .graph.build import build_llm_graph
from .state import DebateState

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _print_debate(result: dict) -> None:
    print(f"\nQuestion: {result['question']}")
    print(f"  for:     {result['positions'].get('for', '')}")
    print(f"  against: {result['positions'].get('against', '')}")

    for entry in result["transcript"]:
        print(f"\n[round {entry.round}] {entry.side} ({entry.stage}):")
        print(f"  {entry.validated_text or '(everything struck for lacking citation)'}")
        if entry.struck_sentences:
            print(f"  struck: {entry.struck_sentences}")

    print(f"\nStopped after round {result['round_count']}: {result['controller_decision']}")
    verdict = result["verdict"]
    print(f"\nVerdict: {verdict['ruling']} (confidence {verdict['confidence']:.2f})")
    print(f"Deciding evidence: {verdict['deciding_evidence']}")
    print(f"Rationale: {verdict['rationale']}")

    print("\nEvidence ledger:")
    for e in result["evidence_ledger"]:
        print(f"  [{e.id}] {e.query}")
        print(f"       -> {e.interpretation}")


def _save_trace(graph, config, debate_id: str) -> Path:
    snapshots = list(graph.get_state_history(config))
    snapshots.reverse()  # chronological order

    trace = []
    for snap in snapshots:
        try:
            values = DebateState.model_validate(snap.values).model_dump(mode="json")
        except ValidationError:
            # The pre-START snapshot has no input applied yet (question is
            # required), so it can't validate -- expected, not a bug.
            values = None
        trace.append({
            "step": snap.metadata.get("step") if snap.metadata else None,
            "next": list(snap.next),
            "values": values,
        })

    out_dir = RUNS_DIR / debate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trace.json"
    out_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    # mock is deliberately excluded here -- it needs a scripted plan (see
    # tests/test_llm_nodes.py), which isn't something a CLI flag can express.
    provider_default = SETTINGS.llm_provider if SETTINGS.llm_provider in ("ollama", "groq") else "ollama"
    parser.add_argument("--provider", default=provider_default, choices=["ollama", "groq"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-rounds", type=int, default=SETTINGS.max_rounds)
    parser.add_argument("--thread-id", default=None)
    args = parser.parse_args()

    if args.provider != "mock" and not SEED_DB_PATH.exists():
        seed_db()

    debate_id = args.thread_id or uuid.uuid4().hex[:12]
    Path(SETTINGS.checkpoint_db_path).parent.mkdir(parents=True, exist_ok=True)

    with SqliteSaver.from_conn_string(SETTINGS.checkpoint_db_path) as checkpointer:
        graph = build_llm_graph(args.provider, model=args.model, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": debate_id}}
        initial = DebateState(question=args.question, max_rounds=args.max_rounds)

        result = graph.invoke(initial, config=config)
        _print_debate(result)

        trace_path = _save_trace(graph, config, debate_id)
        print(f"\nDebate id: {debate_id}")
        print(f"Trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
