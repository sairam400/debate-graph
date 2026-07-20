# Debate Graph

Two advocate agents argue opposite positions on a data question, citing only
real SQL query results, until a judge rules on the evidence. Fourth in a
series about making LLM output over data trustworthy -- this one tests
whether adversarial structure catches things a single agent doesn't.

Work in progress.

- Phase 1: graph skeleton, typed pydantic state, real conditional edges,
  sqlite checkpointing, verified argue/rebut cycle -- all against mock
  nodes, zero LLM calls.
- Phase 2: real `run_sql` (ported, same guardrails) against the seeded
  recommerce db, a real evidence ledger, and a real citation validator that
  strikes uncited (or fabricated-citation) sentences. Positions and the
  judge were still templated placeholders.
- Phase 3: real prompts for `assign_positions`, `advocate_for/against`, and
  `judge`, plus a `get_schema` tool (without it, models were guessing column
  names and failing). Provider factory (`ollama` / `groq` / `mock`) and a
  rate-limited `ChatGroq` wrapper. Verified against the mock provider (zero
  network calls) and against live Ollama (`qwen2.5:3b` -- the pull for the
  spec's actual `qwen2.5:7b` default kept stalling on an unstable
  connection). Real local models are noisy: see KNOWN_ISSUES.md for what
  that noise looked like and why it's the debate structure's job to catch,
  not this project's.

  To try it once you have Ollama running with a model pulled:
  ```
  pip install -r requirements.txt
  python -m src.demo_cli --provider ollama --model qwen2.5:7b --question "Which product category generated the most revenue?"
  ```
- Phase 4 (in progress): 14-question eval set (12 with a ground-truth
  answer, 2 verified genuinely ambiguous -- see `src/eval/dataset.py` for
  the actual numbers behind each), a scorer, the solo-analyst baseline graph,
  and the 2x2 harness (`src/eval/run_eval.py`). Wiring verified end to end
  against the mock provider (`python -m src.eval.run_eval --mock`). Real
  Ollama and Groq runs are next.

This README gets the full problem statement, graph diagram, and results
table once the phase 4 experiment has actually run.
