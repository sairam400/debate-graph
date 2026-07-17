# Debate Graph

Two advocate agents argue opposite positions on a data question, citing only
real SQL query results, until a judge rules on the evidence. Fourth in a
series about making LLM output over data trustworthy -- this one tests
whether adversarial structure catches things a single agent doesn't.

Work in progress. Phase 1 (this commit): graph skeleton with typed pydantic
state, real conditional edges, sqlite checkpointing, and a verified argue/
rebut cycle -- all against mock nodes, zero LLM calls. Real prompts, tools,
and the eval experiment land in later phases; this README gets the full
problem statement, graph diagram, quickstart, and results table once there's
something real to report.
