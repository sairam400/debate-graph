"""Drives the 2x2 experiment: {solo, debate} x {small model, big model} over
the 14-question dataset. Designed to survive being killed and rerun --
results are flushed to results.json after every case, and already-completed
(case, condition, model_label) combos are skipped on the next run. That's a
coarser layer than the debate graph's own langgraph checkpoint (which
resumes a single run mid-flight); this is what makes an overnight Groq run
safe to leave unattended even if the whole process dies.

Usage:
  python -m src.eval.run_eval --mock                       # wiring check, no network
  python -m src.eval.run_eval --small-provider ollama --small-model qwen2.5:7b
  python -m src.eval.run_eval --big-provider groq --big-model llama-3.3-70b-versatile
  python -m src.eval.run_eval                               # both models, both conditions
"""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.checkpoint.sqlite import SqliteSaver

from ..config import SETTINGS
from ..data.seed import DB_PATH as SEED_DB_PATH, build as seed_db
from ..graph.build import build_llm_graph
from ..graph.solo_build import build_solo_graph
from ..state import DebateState, SoloState
from .dataset import CASES
from .scorer import score_case

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "runs" / "eval"
CONDITIONS = ("solo", "debate")
# A single stuck HTTP call defeats "runs unattended" more than a generous
# per-case cap does. Seen in testing: a stalled local-Ollama connection with
# zero CPU activity on either side, well past the point any of the other 27
# cases took. This doesn't kill the underlying blocked thread (Python can't
# force that) -- it just stops waiting on it and lets the harness move on;
# the thread is leaked, not the process.
CASE_TIMEOUT_SECONDS = 600


def _run_key(case_id: str, condition: str, model_label: str) -> str:
    return f"{case_id}::{condition}::{model_label}"


def _load_existing(results_path: Path) -> dict:
    if not results_path.exists():
        return {}
    records = json.loads(results_path.read_text(encoding="utf-8"))
    return {_run_key(r["case_id"], r["condition"], r["model_label"]): r for r in records}


def _save(results_path: Path, records_by_key: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records_by_key.values(), key=lambda r: (r["case_id"], r["condition"], r["model_label"]))
    results_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")


def run_one_case(case: dict, condition: str, provider: str, model_or_plans, model_label: str, checkpoint_path: str) -> dict:
    thread_id = _run_key(case["id"], condition, model_label)
    if provider == "mock":
        model, plan = None, model_or_plans[condition]
    else:
        model, plan = model_or_plans, None

    start = time.monotonic()
    with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer, get_usage_metadata_callback() as cb:
        config = {"configurable": {"thread_id": thread_id}}
        if condition == "debate":
            graph = build_llm_graph(provider, model=model, plan=plan, checkpointer=checkpointer)
            result = graph.invoke(DebateState(question=case["question"], max_rounds=SETTINGS.max_rounds), config=config)
            final_answer = result["verdict"]["final_answer"]
            rounds = result["round_count"]
        else:
            graph = build_solo_graph(provider, model=model, plan=plan, checkpointer=checkpointer)
            result = graph.invoke(SoloState(question=case["question"]), config=config)
            final_answer = result["final_answer"]
            rounds = 1
    wall_time = time.monotonic() - start
    tokens = sum(u.get("total_tokens", 0) for u in cb.usage_metadata.values())

    score = score_case(case, final_answer)
    return {
        **score,
        "condition": condition,
        "model_label": model_label,
        "provider": provider,
        "model": model,
        "tokens": tokens,
        "wall_time_seconds": round(wall_time, 3),
        "rounds": rounds,
    }


def _failure_record(case, condition, provider, model_or_plans, model_label, note, wall_time_seconds, **extra) -> dict:
    model = None if provider == "mock" else model_or_plans
    return {
        "case_id": case["id"],
        "correct": False,
        "final_answer": note,
        "gold_answer": case["gold_answer"],
        "condition": condition,
        "model_label": model_label,
        "provider": provider,
        "model": model,
        "tokens": 0,
        "wall_time_seconds": wall_time_seconds,
        "rounds": 0,
        **extra,
    }


def _run_one_case_with_timeout(case, condition, provider, model_or_plans, model_label, checkpoint_path) -> dict:
    # Deliberately not a `with` block: ThreadPoolExecutor.__exit__ calls
    # shutdown(wait=True), which would block right back on the same stuck
    # thread we're trying to stop waiting for. shutdown(wait=False) here
    # lets this function return immediately on timeout; the thread itself
    # is leaked (Python can't force-kill a thread), not the process.
    #
    # Also catches any other exception the worker raises (e.g. httpx read
    # timeouts from the per-request ChatOllama timeout, a Groq error that
    # slips past the rate-limit wrapper's own retries) -- an unhandled
    # exception here previously crashed the whole multi-hour run over a
    # single bad case, discovered by that exact thing happening live.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        run_one_case, case, condition, provider, model_or_plans, model_label, checkpoint_path
    )
    try:
        result = future.result(timeout=CASE_TIMEOUT_SECONDS)
        executor.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        return _failure_record(
            case, condition, provider, model_or_plans, model_label,
            note=f"(timed out after {CASE_TIMEOUT_SECONDS}s)",
            wall_time_seconds=CASE_TIMEOUT_SECONDS, timed_out=True,
        )
    except Exception as exc:
        executor.shutdown(wait=False)
        return _failure_record(
            case, condition, provider, model_or_plans, model_label,
            note=f"(errored: {type(exc).__name__}: {exc})",
            wall_time_seconds=0.0, errored=True,
        )


def run_experiment(models: dict, cases=CASES, conditions=CONDITIONS, results_path: Path = None) -> list:
    """models: {model_label: (provider, model_name)}."""
    results_path = results_path or (RESULTS_DIR / "results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(results_path.parent / "checkpoints.sqlite")
    by_key = _load_existing(results_path)

    if any(provider != "mock" for provider, _ in models.values()) and not SEED_DB_PATH.exists():
        seed_db()

    for model_label, (provider, model_or_plans) in models.items():
        for condition in conditions:
            for case in cases:
                key = _run_key(case["id"], condition, model_label)
                existing = by_key.get(key)
                # A timed-out or errored case isn't "done" -- it's exactly
                # the kind of transient failure (a stuck connection, a Groq
                # tokens-per-day limit that clears in a few minutes) a rerun
                # is supposed to recover from. Only a real completed result
                # counts toward skip-on-resume.
                if existing is not None and not existing.get("timed_out") and not existing.get("errored"):
                    continue
                record = _run_one_case_with_timeout(case, condition, provider, model_or_plans, model_label, checkpoint_path)
                by_key[key] = record
                _save(results_path, by_key)
                if record.get("timed_out"):
                    status = "TIMEOUT"
                elif record.get("errored"):
                    status = "ERROR"
                else:
                    status = "PASS" if record["correct"] else "FAIL"
                print(f"{key}: {status} "
                      f"({record['tokens']} tokens, {record['wall_time_seconds']}s, {record['rounds']} rounds)")

    return list(by_key.values())


def summarize(records: list) -> str:
    groups = {}
    for r in records:
        key = (r["model_label"], r["condition"])
        groups.setdefault(key, []).append(r)

    lines = ["| model | condition | accuracy | avg tokens | avg wall time (s) | avg rounds |",
             "|---|---|---|---|---|---|"]
    for (model_label, condition), rows in sorted(groups.items()):
        n = len(rows)
        accuracy = sum(r["correct"] for r in rows) / n
        avg_tokens = sum(r["tokens"] for r in rows) / n
        avg_wall = sum(r["wall_time_seconds"] for r in rows) / n
        avg_rounds = sum(r["rounds"] for r in rows) / n
        lines.append(
            f"| {model_label} | {condition} | {accuracy:.0%} | {avg_tokens:.0f} | {avg_wall:.1f} | {avg_rounds:.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="wiring check: both conditions against a scripted mock model")
    parser.add_argument("--small-provider", default="ollama")
    parser.add_argument("--small-model", default=SETTINGS.ollama_model)
    parser.add_argument("--big-provider", default="groq")
    parser.add_argument("--big-model", default=SETTINGS.groq_model)
    parser.add_argument("--only", choices=["small", "big"], default=None)
    args = parser.parse_args()

    if args.mock:
        from .mock_plans import MOCK_MODELS
        models = MOCK_MODELS
    else:
        models = {}
        if args.only != "big":
            models["small"] = (args.small_provider, args.small_model)
        if args.only != "small":
            models["big"] = (args.big_provider, args.big_model)

    records = run_experiment(models)
    table = summarize(records)
    print("\n" + table)
    (RESULTS_DIR / "summary.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
