# Known issues

**A worker exception crashed the entire eval run over a single case.** The
per-case `ThreadPoolExecutor` timeout only caught `concurrent.futures.
TimeoutError` (the case hanging past its budget) -- it did not catch an
exception the worker actually *raised*, which `future.result()` re-raises
in the caller. This happened live: the 180s `ChatOllama` request timeout
(below) fired correctly and raised `httpx.ReadTimeout`, which propagated
uncaught through the whole run and killed the process mid-experiment, 10 of
14 debate cases in. Fixed by also catching `Exception` in
`_run_one_case_with_timeout` and recording it as an `errored` case instead
of letting it propagate -- covered by
`test_worker_exception_is_recorded_not_raised` in `tests/test_run_eval.py`.
Discovered this way rather than in review, which is itself the argument for
why the "verify on real infrastructure before committing" step of this
project's process matters -- a mocked test never would have hit this, since
`MockChatModel` doesn't make real HTTP calls.

**A single stuck case had no ceiling, so it could block the whole eval run
even with the request timeout below.** The per-request `ChatOllama` timeout
bounds one HTTP call, but a case can make many calls (multiple rounds x
both sides x multiple tool calls) with no overall cap, and that's exactly
what happened in testing -- a debate case sat idle for 30+ minutes with
zero CPU on both the client and `ollama serve` before being killed manually.
Fixed with a per-case wall-clock cap (`CASE_TIMEOUT_SECONDS`, 600s) in
`eval/run_eval.py`, using a `ThreadPoolExecutor` so the harness can abandon
a stuck case and move on. The blocked thread itself is leaked (Python can't
force-kill a thread) and lingers until the whole process eventually exits --
that's an accepted cost, since the property that actually matters for an
unattended run is that one bad case can't stop the other 27 from completing,
not that resources are pristine. `shutdown(wait=False)` in that function is
required for the same reason: the default `wait=True` would block on exactly
the same stuck thread the timeout was supposed to stop waiting for.

**`ChatOllama` had no request timeout, so a stalled connection hung forever.**
During the real phase 4 eval run (qwen2.5:3b, debate condition), the process
went idle -- zero CPU on both the client and the `ollama serve` process --
partway through a case and never recovered; had to be killed manually.
`ChatOllama` doesn't expose a timeout parameter directly, but accepts
`client_kwargs`, which passes through to the underlying `ollama.Client`'s
httpx client. Fixed by passing `client_kwargs={"timeout":
OLLAMA_TIMEOUT_SECONDS}` (default 180s) in `providers/factory.py`. A timeout
turns a silent hang into a real, catchable exception -- which for an
unattended run matters more than getting the number exactly right.

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
