# Debate Graph

Two advocate agents argue opposite positions on a data question, citing only
real SQL query results, until a judge rules on the evidence. Fourth in a
series about making LLM output over data trustworthy -- the first three dealt
with retrieval, tool-use, and a separate verifier that recomputes citations.
This one asks whether adversarial structure -- one model checking another's
work in real time, not after the fact -- catches things either a single
agent or a bolt-on verifier doesn't.

Hard constraint: zero API spend. Every real run uses either Ollama locally
(free) or Groq's free tier. Development runs against a scripted mock
provider that makes zero network calls. No Anthropic or OpenAI key
anywhere in this project.

## The graph

```
                 START
                   |
                   v
           assign_positions            (peeks schema/data once, frames
                   |                     a FOR claim and an AGAINST stance)
                   v
    +------> advocate_for -----> validate_for
    |         (run_sql, cite [Ex])   (strikes uncited/fabricated citations)
    |                                       |
    |                                       v
    |         advocate_against -----> validate_against
    |         (run_sql, cite [Ex])          |
    |                                       v
    |                                  controller
    |          (continue / stop_max_rounds / stop_concession /
    |            stop_no_new_evidence -- deterministic, no LLM call)
    |                                       |
    +--------------- continue --------------+
                                             | stop
                                             v
                                          judge          -> END
                            (validated transcript + evidence ledger only;
                             ruling, confidence, deciding evidence,
                             final_answer -- can rule "unsettled")
```

The argue/rebut loop is a real graph cycle (`advocate_for` gets invoked
again via the conditional edge after `controller` says `continue`), not
unrolled per round -- verified in `tests/test_graph_cycle.py` by checking
`advocate_for`/`advocate_against` run multiple times across a multi-round
debate. SQLite checkpointing means a debate can resume mid-flight, which is
also what makes an unattended eval run survivable if it's interrupted.

**Solo baseline** (the other half of the experiment) is a separate, minimal
3-node graph -- `analyst -> validate -> finalize` -- reusing the same
`run_sql`/`get_schema` tools and the same citation validator, so the 2x2
experiment compares graph *shape*, not tooling.

## Quickstart (Ollama only, no API key)

```
pip install -r requirements.txt
ollama pull qwen2.5:7b        # or a smaller qwen2.5 variant -- see gaps below
python -m src.demo_cli --provider ollama --model qwen2.5:7b \
  --question "Which product category generated the most revenue?"
```

Prints the positions, each round's argument (with anything the citation
validator struck flagged separately), the verdict, and the evidence ledger,
then saves the full graph execution trace to `runs/<debate_id>/trace.json`.

Tests (zero network calls, scripted mock provider):
```
python -m unittest discover -s tests
```

Full 2x2 eval harness:
```
python -m src.eval.run_eval --mock                                          # wiring check
python -m src.eval.run_eval --small-provider ollama --small-model qwen2.5:7b --only small
python -m src.eval.run_eval --big-provider groq --big-model llama-3.3-70b-versatile --only big
```
Results land in `runs/eval/results.json`, resumable -- a killed process or a
rate-limited case just gets retried on the next invocation, not restarted
from scratch.

## The results

14 questions over a seeded recommerce SQLite db (~5k order line items): 12
with a single correct answer, 2 written and verified against the actual data
to be genuinely ambiguous (see `src/eval/dataset.py` for the real numbers
behind each -- e.g. Electronics leads revenue but Sporting Goods has the
lower return rate, so "which category is healthier" has no single right
answer). Small model: `qwen2.5:3b` (see gaps below for why not the spec's
`qwen2.5:7b`). Big model: `llama-3.3-70b-versatile` via Groq's free tier.

| model | condition | accuracy | avg tokens | avg wall time (s) | avg rounds |
|---|---|---|---|---|---|
| big   | debate | 29% | 4380 | 21.9  | 0.6 |
| big   | solo   | 50% | 2196 | 7.1   | 1.0 |
| small | debate | 21% | 7067 | 180.5 | 1.5 |
| small | solo   | 14% | 1955 | 273.9 | 1.0 |

**The big/debate row is not a clean number.** Groq's free tier has a
100,000-tokens-per-day cap, and this run hit it partway through -- 7 of the
14 big/debate cases errored out with no answer at all (counted as
incorrect above, which is the conservative/honest choice, but it means
"29%" is really "4 real passes out of 7 cases that got a real attempt,
plus 7 cases that never ran." Excluding the rate-limited cases entirely:
**big/debate on the 7 cases that actually completed scored 57%** (4/7).
Neither number should be taken as a clean read on the 70b model's real
debate accuracy -- the honest conclusion is that this run didn't get enough
big/debate data to say, not that debate hurt the big model.

**Small model, uncontaminated by any rate limit, is the clean comparison:
14% solo vs. 21% debate.** Debate structure helped the small model here.
Whether it helps big models too is genuinely unanswered by this run --
answering it needs a rerun of just the big/debate condition once the daily
token window clears, which the harness supports for free (`--only big`,
same results file, already-correct cases are skipped).

## Design decisions

**The citation validator strikes fabricated citations, not just missing
ones.** A naive check ("does this sentence contain something that looks
like `[Ex]`") would let a model cite `[E99]` that was never gathered and
still pass. `validate_argument` checks the id actually exists in the
evidence ledger. This mattered in practice: a live Ollama run had an
advocate cite a nonexistent evidence id after a failed query, and it got
struck correctly.

**The controller's stop decision is deterministic, not another LLM call.**
`max_rounds` / concession detected / no new evidence this round -- three
checkable rules, not a model's judgment about whether the debate feels
finished. Auditable, and it means the graph's shape doesn't depend on the
model's mood.

**The judge's `final_answer` field is a late addition, not in the original
design.** Grading debate output against the same gold answers as the solo
baseline needs a direct answer, not just "for beat against" -- ruling alone
doesn't say *what* the answer is when the debate concludes "against" (a
rival, unstated claim) or "unsettled." Added once the eval harness made the
gap obvious, not anticipated up front.

**What didn't work: assuming a stable connection and no request timeout
were safe defaults.** Three real bugs, found only by actually running this
against live infrastructure instead of mocking or reviewing it:
`ChatOllama` has no request timeout by default, so a stalled connection
hung forever; the per-case timeout wrapper only caught the *timeout*
exception, not a real exception the worker raised (which is exactly what
happened once the Ollama timeout fix started firing for real, and it took
the whole multi-hour run down with it); and Groq's tokens-per-day limit is
a different failure mode than RPM, with no fixed reset time, so blind
exponential backoff either gave up too early or waited too long -- fixed by
parsing the "try again in Xm Ys" hint Groq's own error message provides.
Full details in `KNOWN_ISSUES.md`. None of these would have surfaced from
the mock-provider test suite alone, which is the actual argument for why
"verify on real infrastructure before committing" was worth the time it cost.

## What this proves and doesn't

**Proves:** the graph mechanics work end to end on real infrastructure --
cycles, conditional edges, checkpointing, citation validation catching a
real hallucinated citation, judge fallback on malformed output -- and, for
the one clean comparison this run produced, debate structure improved a
small local model's accuracy on this question set (14% to 21%).

**Doesn't prove:** that debate helps more than it costs. Debate used
3.6x the tokens and ~15x the wall time of solo for the small model, for a
7-point accuracy gain -- expensive per point. It doesn't answer the
headline question (does debate help small models more than big ones) for
the big-model side, because that comparison never got a full, clean run.
It also doesn't establish that debate generalizes past 12 graded questions
on one seeded dataset -- an *n* of 12-14 per cell is enough to see a signal,
not enough to trust the exact percentages.

## Known limitations

See `KNOWN_ISSUES.md` for the complete list. Headline items: `qwen2.5:7b`
(the spec's actual default) was never run live, due to a genuinely unstable
connection during setup -- `qwen2.5:3b` stood in for the eval run instead,
which is a materially weaker model, and `qwen2.5:0.5b` (tried first, even
smaller) couldn't use tools at all; the big/debate results are incomplete
for the reason described above; and `assign_positions` fell back to its
generic template in every live small-model run so far, meaning its JSON
output never actually parsed against a 3b model -- untested against 7b or
the 70b model, where it may well work fine.
