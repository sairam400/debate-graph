"""Strikes any argument sentence that carries no citation, or that cites an
evidence id that doesn't actually exist in the ledger -- a naive regex check
for "[E\\d+] somewhere in the sentence" would let a model cite a fabricated
[E99] and slip past. Sentence splitting is a plain punctuation-boundary regex,
not a real tokenizer: it does not handle abbreviations, and a decimal like
"$12.5 million" only survives because there's no space directly after the
period. Good enough for the short, plain-prose arguments this graph
generates; flagged in KNOWN_ISSUES as a real limitation, not a hidden one.
"""
import re

from ..state import EvidenceEntry

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITATION = re.compile(r"\[E\d+\]")


def validate_argument(raw_text: str, ledger: list[EvidenceEntry]) -> dict:
    known_ids = {e.id for e in ledger}
    sentences = [s for s in _SENTENCE_SPLIT.split(raw_text.strip()) if s]

    kept, struck, citations = [], [], []
    for sentence in sentences:
        found = [m.strip("[]") for m in _CITATION.findall(sentence)]
        valid = [cid for cid in found if cid in known_ids]
        if valid:
            kept.append(sentence)
            citations.extend(valid)
        else:
            struck.append(sentence)

    return {
        "validated_text": " ".join(kept),
        "citations": sorted(set(citations), key=lambda cid: int(cid[1:])),
        "struck_sentences": struck,
    }
