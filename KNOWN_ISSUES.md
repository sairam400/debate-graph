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

**qwen2.5:3b is inconsistent at actually using the tools it's given, across
runs of the identical prompt.** Verified live against Ollama (qwen2.5:3b):
the full graph runs end to end with no crashes -- checkpointing, tracing,
citation striking, and the judge's parse-failure fallback all behave
correctly against a real model's real (sometimes broken) output. In one
isolated single-turn test the model called `get_schema`, then wrote a
correct query, and got the right answer (Electronics, $218,485.55). In two
separate full graph runs of the same question immediately after, both
advocates instead hallucinated that `order_items` has no `quantity` column
-- which is false, and which `get_schema` would have told them otherwise --
and neither ever produced a real citation, so the debate ended
`stop_no_new_evidence` both times. Same prompt structure, different
outcomes: this reads as real local-inference nondeterminism (quantized
model, batching-order effects) rather than a bug in the tool-loop wiring,
since the isolated test proves the wiring itself works. qwen2.5:0.5b is
worse: it never emitted a single tool call in testing, just answered from
guesswork -- too small for reliable tool use. **This is exactly the kind of
noise the debate structure and citation validator exist to catch**, and
qwen2.5:7b (the project's actual spec default, not yet tested live -- see
below) should be meaningfully more reliable than 3b.

**qwen2.5:7b, the project's specified default small model, has not been
verified live yet.** This environment's network had heavy packet loss
during setup, so `ollama pull qwen2.5:7b` repeatedly stalled and restarted
partway through; qwen2.5:0.5b and qwen2.5:3b were used instead to get a real
end-to-end run without waiting indefinitely on an unstable link. Pulling
qwen2.5:7b properly (ideally on a more stable connection) and re-running
`python -m src.demo_cli --provider ollama --model qwen2.5:7b` is worth doing
before the phase 4 experiment, since 7b is the model the results table
should actually report for "small model."

**assign_positions falls back to its generic template in every live Ollama
run so far** (`An answer to: <question>` / `A different answer to: ...`),
meaning its JSON output didn't parse on any of these small-model attempts.
Not yet diagnosed further -- see the asymmetric-retry note below.

**assign_positions and judge JSON-parsing retries are asymmetric.** `judge`
retries once with the parse error fed back to the model
(`providers/llm_json.complete_json`). `assign_positions` does not retry --
it runs an optional tool-call turn first, so a naive retry would re-run that
turn too; on a parse failure it falls back to a deterministic templated
position pair instead. Worth revisiting once real-model output shows how
often that fallback actually fires.

**Checkpoint serialization warns on the custom pydantic types.** langgraph's
default msgpack serializer doesn't recognize `DebateState`/`TranscriptEntry`/
`EvidenceEntry` as registered types yet, so every checkpoint write/read logs
a deprecation warning ("will be blocked in a future version"). Harmless
today, but needs an explicit `allowed_msgpack_modules` (or a custom serde)
before upgrading langgraph past whatever version makes that a hard error.
