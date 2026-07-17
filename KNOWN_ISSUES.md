# Known issues

**`assign_positions` and `judge` are still templated, not real prompts.**
They return fixed-shape placeholder text. `advocate_for/against` are real as
of phase 2 (real `run_sql` against the seeded db, real evidence ledger) but
still pick their query from a small fixed rotation rather than a model
deciding what to ask -- that's the one piece phase 3 replaces. Every
argument also includes one deliberately uncited filler sentence so the
citation validator has something real to strike in the end-to-end tests, not
just in validator.py's own unit tests.

**Sentence splitting in the citation validator is a punctuation regex, not a
real tokenizer.** Works fine for the plain, short prose these nodes generate
today; would mis-split text with abbreviations or unusual punctuation. See
`src/graph/validator.py` docstring.

**Checkpoint serialization warns on the custom pydantic types.** langgraph's
default msgpack serializer doesn't recognize `DebateState`/`TranscriptEntry`/
`EvidenceEntry` as registered types yet, so every checkpoint write/read logs
a deprecation warning ("will be blocked in a future version"). Harmless
today, but needs an explicit `allowed_msgpack_modules` (or a custom serde)
before upgrading langgraph past whatever version makes that a hard error.
