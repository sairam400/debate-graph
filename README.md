# Debate Graph

Two advocate agents argue opposite positions on a data question, citing only
real SQL query results, until a judge rules on the evidence. Fourth in a
series about making LLM output over data trustworthy -- this one tests
whether adversarial structure catches things a single agent doesn't.

Work in progress.

- Phase 1: graph skeleton, typed pydantic state, real conditional edges,
  sqlite checkpointing, verified argue/rebut cycle -- all against mock
  nodes, zero LLM calls.
- Phase 2 (this commit): real `run_sql` (ported, same guardrails) against
  the seeded recommerce db, a real evidence ledger, and a real citation
  validator that strikes uncited (or fabricated-citation) sentences.
  Positions and the judge are still templated placeholders. Still zero LLM
  calls -- `python -m unittest discover -s tests` runs entirely offline.

Real prompts and the eval experiment land in later phases; this README gets
the full problem statement, graph diagram, quickstart, and results table
once there's something real to report.
