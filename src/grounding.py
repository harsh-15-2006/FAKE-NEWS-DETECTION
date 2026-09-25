"""
GROUNDING VALIDATOR — the component that makes the LLM stage defensible.

This is a deterministic post-check, not a prompt. Careful prompting is a hope;
this is an enforced guarantee.

Three checks, any failure forces NOT_IN_CONTEXT:

  1. SPAN CHECK      the quoted_span must actually occur in a cited passage.
                     If the model invented, paraphrased or "improved" the
                     quote, the span will not be found and the output dies.
  2. CITATION CHECK  every cited passage id must exist in the evidence pack.
                     A model citing [PASSAGE 7] when only 0-2 were supplied is
                     fabricating a source.
  3. WARRANT CHECK   a SUPPORT or CONTRADICT stance with zero citations is an
                     unwarranted assertion.

Every rejection is recorded and surfaced in the dashboard, because a caught
hallucination is evidence the safeguard works, not something to hide.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .models import EvidencePack, GroundingResult, LLMStance, Stance

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)


def normalize(text: str) -> str:
    """Case-fold, strip punctuation, collapse whitespace, NFKC-normalise.

    Deliberately forgiving about formatting and unforgiving about content: it
    lets through a changed comma, it does not let through a changed word.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def validate(
    llm: LLMStance | None,
    evidence: EvidencePack,
    cfg: dict[str, Any],
) -> GroundingResult:
    gcfg = cfg.get("grounding", {})
    if not gcfg.get("enabled", True):
        return GroundingResult(
            passed=True,
            reasons=["grounding validator disabled by config"],
            enforced_stance=llm.stance if llm else Stance.NOT_IN_CONTEXT,
        )

    if llm is None:
        return GroundingResult(
            passed=False,
            reasons=["no LLM output to validate"],
            enforced_stance=Stance.NOT_IN_CONTEXT,
        )

    # NOT_IN_CONTEXT needs no evidence to back it - it is the abstention.
    if llm.stance == Stance.NOT_IN_CONTEXT:
        return GroundingResult(
            passed=True,
            reasons=["abstention requires no citation"],
            enforced_stance=Stance.NOT_IN_CONTEXT,
        )

    strictness = gcfg.get("strictness", "normalized")
    min_span = int(gcfg.get("min_span_chars", 12))
    reasons: list[str] = []

    valid_ids = {p.pid for p in evidence.passages}

    # --- CHECK 2: cited ids must exist -------------------------------- #
    if not llm.cited_passage_ids:
        reasons.append("WARRANT FAIL: stance asserted with zero citations")
    unknown = [i for i in llm.cited_passage_ids if i not in valid_ids]
    if unknown:
        reasons.append(
            f"CITATION FAIL: cited passage ids {unknown} do not exist "
            f"(supplied: {sorted(valid_ids)})"
        )

    # --- CHECK 1: the span must be in a cited passage ------------------ #
    span = (llm.quoted_span or "").strip()
    if not span:
        reasons.append("SPAN FAIL: no quoted_span returned")
    elif len(span) < min_span:
        reasons.append(
            f"SPAN FAIL: quoted_span is {len(span)} chars, "
            f"below min_span_chars={min_span} - proves nothing"
        )
    else:
        cited = [p for p in evidence.passages if p.pid in llm.cited_passage_ids]
        found = False
        for p in cited:
            if strictness == "exact":
                found = span in p.text
            else:
                found = normalize(span) in normalize(p.text)
            if found:
                break
        if not found:
            preview = span[:80] + ("..." if len(span) > 80 else "")
            reasons.append(
                f'SPAN FAIL: quoted_span "{preview}" does not occur in any '
                f"cited passage - the model did not copy it from the source"
            )

    if reasons:
        return GroundingResult(
            passed=False, reasons=reasons, enforced_stance=Stance.NOT_IN_CONTEXT
        )

    return GroundingResult(
        passed=True,
        reasons=[f"span verified verbatim in passage(s) {llm.cited_passage_ids}"],
        enforced_stance=llm.stance,
    )
