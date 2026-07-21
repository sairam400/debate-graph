"""Grades one final_answer string against one dataset.py case. Same scorer
for both experiment conditions (solo and debate) -- both produce a plain
final_answer string, solo directly, debate via the judge's "final_answer"
field -- so nothing here needs to know which condition produced the answer.

Matching is deliberately lenient (substring for strings, first-number-found
for numerics): the point is grading whether the *content* is right, not
enforcing an exact output format on a model we don't control the phrasing of.
"""
import re

_NUMBER = re.compile(r"-?[\d,]+\.?\d*")

_UNSETTLED_MARKERS = (
    "cannot be determined",
    "cannot settle",
    "can't settle",
    "data cannot",
    "unsettled",
    "insufficient data",
    "cannot be resolved",
    "no clear answer",
)


def _extract_numbers(text: str) -> list[float]:
    values = []
    for match in _NUMBER.finditer(text.replace("$", "")):
        try:
            values.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return values


def _numeric_match(final_answer: str, gold: float, tolerance: float) -> bool:
    # Any number in the text within tolerance counts -- a model's answer
    # sentence often contains other numbers (dates, evidence ids, an
    # intermediate figure), so requiring the *first* or *only* number to be
    # the right one is too brittle. This is scoring against a known gold
    # value, not extracting an answer blind, so the leniency doesn't let
    # wrong answers slip through: a text with no number close to gold still
    # fails.
    return any(abs(v - gold) <= tolerance for v in _extract_numbers(final_answer))


def _string_match(final_answer: str, gold: str) -> bool:
    return gold.strip().lower() in final_answer.strip().lower()


def _is_unsettled(final_answer: str) -> bool:
    text = final_answer.strip().lower()
    return any(marker in text for marker in _UNSETTLED_MARKERS)


def score_case(case: dict, final_answer: str) -> dict:
    answer_type = case["answer_type"]

    if answer_type == "unsettled":
        correct = _is_unsettled(final_answer)
    elif answer_type == "numeric":
        correct = _numeric_match(final_answer, case["gold_answer"], case.get("tolerance", 0))
    elif answer_type == "string":
        correct = _string_match(final_answer, case["gold_answer"])
    else:
        raise ValueError(f"unknown answer_type: {answer_type!r}")

    return {
        "case_id": case["id"],
        "correct": correct,
        "final_answer": final_answer,
        "gold_answer": case["gold_answer"],
    }
