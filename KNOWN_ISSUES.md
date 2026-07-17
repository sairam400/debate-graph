# Known issues

**Phase 1 nodes are mocked.** `assign_positions`, `advocate_for/against`,
`validate_for/against`, and `judge` currently return deterministic canned
text -- no LLM, no `run_sql`. Only `controller` is real. This proves the
graph's shape (cycle + conditional edges + checkpointing) works before any
model or database is involved. Real nodes land in phase 2/3.

**Checkpoint serialization warns on the custom pydantic types.** langgraph's
default msgpack serializer doesn't recognize `DebateState`/`TranscriptEntry`/
`EvidenceEntry` as registered types yet, so every checkpoint write/read logs
a deprecation warning ("will be blocked in a future version"). Harmless
today, but needs an explicit `allowed_msgpack_modules` (or a custom serde)
before upgrading langgraph past whatever version makes that a hard error.
