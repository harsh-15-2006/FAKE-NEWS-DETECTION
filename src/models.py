"""
Typed objects passed between pipeline layers.

Each layer returns one of these, so every layer is independently testable and
independently demoable. No layer reaches into another layer's internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """The four terminal states. INSUFFICIENT_EVIDENCE is a first-class
    result, not a failure — it is the honest answer when retrieval comes
    back thin, and it is the state a trained classifier cannot produce."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ESCALATE = "ESCALATE"


class Stance(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    NOT_IN_CONTEXT = "NOT_IN_CONTEXT"


@dataclass
class LangResult:
    raw: str
    native_script: str            # input rendered in its own script
    lang_code: str                # ta | hi | en
    pivot_en: str                 # English rendering used for retrieval/LLM
    was_romanized: bool
    script_counts: dict[str, int]
    translit_method: str          # llm | rule | none
    translate_method: str         # llm | passthrough | none
    notes: list[str] = field(default_factory=list)


@dataclass
class Claim:
    idx: int
    text_native: str
    text_en: str
    lang_code: str
    checkworthy: float
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signals:
    """The signal bus. Computed once before retrieval, consumed twice:
    by the search router (strategy selection) and by the score engine
    (fusion). In the original hand drawing these died at the router."""

    rule_score: float
    rule_hits: list[str]
    ml_score: float
    ml_basis: str
    category: str
    category_confidence: float
    policy: dict[str, Any]


@dataclass
class Passage:
    pid: int
    text: str
    publisher: str
    url: str
    rating: str
    claim_date: str | None
    review_date: str | None
    language: str
    source: str                  # api:ta | api:en | offline | sample
    resurfaced: bool = False
    resurface_gap_days: int | None = None
    synthetic: bool = False      # True for the labelled sample corpus


@dataclass
class EvidencePack:
    passages: list[Passage]
    strategy_used: str
    paths_queried: list[str]
    n_unique: int
    any_resurfaced: bool
    synthetic_only: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class LLMStance:
    stance: Stance
    cited_passage_ids: list[int]
    quoted_span: str
    reasoning: str
    raw: str
    model: str


@dataclass
class GroundingResult:
    passed: bool
    reasons: list[str]
    enforced_stance: Stance      # what the system will actually use


@dataclass
class ClaimResult:
    claim: Claim
    signals: Signals
    evidence: EvidencePack
    llm: LLMStance | None
    grounding: GroundingResult | None
    verdict: Verdict
    confidence: float
    gate_fired: str
    reasoning: str
    score_breakdown: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        if self.llm is not None:
            d["llm"]["stance"] = self.llm.stance.value
        if self.grounding is not None:
            d["grounding"]["enforced_stance"] = self.grounding.enforced_stance.value
        return d


@dataclass
class Report:
    input_text: str
    language: LangResult
    claims: list[ClaimResult]
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_text": self.input_text,
            "language": asdict(self.language),
            "claims": [c.to_dict() for c in self.claims],
            "aggregate": self.aggregate,
        }
