"""Typed state for the debate graph. Every node reads/returns partial updates
against this schema; langgraph validates them through pydantic on each
super-step, so a node that returns a malformed field fails loudly instead of
corrupting the checkpoint."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Side = Literal["for", "against"]
ControllerDecision = Literal[
    "continue", "stop_max_rounds", "stop_concession", "stop_no_new_evidence"
]


class EvidenceEntry(BaseModel):
    id: str
    query: str
    result: str
    interpretation: str
    round: int
    side: Side


class TranscriptEntry(BaseModel):
    round: int
    side: Side
    stage: Literal["argument", "rebuttal"]
    raw_text: str
    validated_text: str
    citations: list[str] = Field(default_factory=list)
    struck_sentences: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    ruling: str
    confidence: float
    deciding_evidence: list[str] = Field(default_factory=list)
    rationale: str


class DebateState(BaseModel):
    question: str
    positions: dict[Side, str] = Field(default_factory=dict)
    evidence_ledger: list[EvidenceEntry] = Field(default_factory=list)
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    round_count: int = 0
    max_rounds: int = 3
    controller_decision: Optional[ControllerDecision] = None
    verdict: Optional[Verdict] = None
