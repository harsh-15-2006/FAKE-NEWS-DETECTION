"""
SCORE ENGINE — gates first, fusion second.

Gate order matters and is the safety property of the whole system. The
evidence gates run BEFORE the fused score is consulted at all, so a high rule
score can never on its own produce a REFUTED verdict:

  1. category in never_auto_verdict      -> ESCALATE
  2. n_sources < evidence_min_sources    -> INSUFFICIENT_EVIDENCE
  3. LLM abstained (NOT_IN_CONTEXT)      -> INSUFFICIENT_EVIDENCE
  4. grounding validator rejected output -> INSUFFICIENT_EVIDENCE
  5. sources conflict above threshold    -> ESCALATE
  6. otherwise                           -> fused score decides

Fusion weights (all configurable): llm 0.45, retrieval 0.30, ml 0.15,
rule 0.10. Evidence dominates by design. Rule and ML together are 25% - they
are priors that break ties, never the verdict.
"""

from __future__ import annotations

from typing import Any

from .models import (
    Claim, ClaimResult, EvidencePack, GroundingResult, LLMStance,
    Signals, Stance, Verdict,
)

# Publisher tiers. IFCN-certified Indian fact-checkers and established
# international desks weigh more than an unrecognised source. Override with
# scoring.publisher_tiers in settings.yaml.
DEFAULT_TIERS = {
    "alt news": 1.0, "altnews": 1.0,
    "boom": 1.0, "boomlive": 1.0, "boom live": 1.0,
    "factly": 1.0, "newschecker": 1.0, "vishvas news": 0.95,
    "the quint": 0.95, "webqoof": 0.95, "india today": 0.9,
    "afp": 1.0, "reuters": 1.0, "pti": 0.9, "pib": 0.9,
    "snopes": 0.95, "politifact": 0.95, "factcheck.org": 0.95,
    "full fact": 0.95, "logically": 0.85,
}

_REFUTING_RATINGS = {
    "false", "fake", "incorrect", "misleading", "mostly false", "pants on fire",
    "altered", "miscaptioned", "no evidence", "unproven", "distorted",
    "manipulated", "fabricated", "not true", "hoax", "satire",
}
_SUPPORTING_RATINGS = {
    "true", "correct", "mostly true", "accurate", "verified", "genuine",
}


def publisher_tier(name: str, cfg: dict[str, Any]) -> float:
    tiers = {**DEFAULT_TIERS, **(cfg.get("scoring", {}).get("publisher_tiers") or {})}
    low = (name or "").casefold().strip()
    for key, val in tiers.items():
        if key in low:
            return float(val)
    return 0.6  # unrecognised but present


def rating_direction(rating: str) -> int:
    """-1 refutes, +1 supports, 0 unclear. Ratings are free text per publisher,
    so this maps them onto a common scale rather than trusting the string."""
    low = (rating or "").casefold().strip()
    if not low:
        return 0
    for token in _REFUTING_RATINGS:
        if token in low:
            return -1
    for token in _SUPPORTING_RATINGS:
        if token in low:
            return 1
    return 0


def evidence_direction(evidence: EvidencePack, cfg: dict[str, Any]
                       ) -> tuple[float, float, dict[str, float]]:
    """Returns (direction, conflict, detail).
    direction: -1..+1 weighted by publisher tier. Negative = refuting.
    conflict : 0..1, how split the sources are."""
    refute_w = support_w = 0.0
    for p in evidence.passages:
        tier = publisher_tier(p.publisher, cfg)
        d = rating_direction(p.rating)
        if d < 0:
            refute_w += tier
        elif d > 0:
            support_w += tier
    total = refute_w + support_w
    if total == 0:
        return 0.0, 0.0, {"refute_weight": 0.0, "support_weight": 0.0}
    direction = (support_w - refute_w) / total
    conflict = min(refute_w, support_w) / total * 2
    return (
        round(direction, 3),
        round(conflict, 3),
        {"refute_weight": round(refute_w, 3), "support_weight": round(support_w, 3)},
    )


SYNTHETIC_PENALTY = 0.7


def retrieval_strength(evidence: EvidencePack, policy: dict[str, Any]) -> float:
    """Meeting the category's source minimum counts as full strength; labelled
    synthetic sample data is discounted and says so on screen."""
    need = max(1, int(policy.get("evidence_min_sources", 2)))
    base = min(1.0, evidence.n_unique / need)
    if evidence.synthetic_only:
        base *= SYNTHETIC_PENALTY
    return round(base, 3)


def rating_confidence(evidence: EvidencePack, direction: float,
                      cfg: dict[str, Any]) -> float:
    """Confidence carried by the published ratings themselves.

    Used in the LLM slot when no grounded model stance is available. This is
    not a fallback guess: a textualRating from an IFCN-certified fact-checker
    IS a verdict, reached by a human who did the work. Weighting it by
    publisher tier and by how one-sided the ratings are keeps that honest.
    """
    rated = [p for p in evidence.passages if rating_direction(p.rating) != 0]
    if not rated:
        return 0.0
    mean_tier = sum(publisher_tier(p.publisher, cfg) for p in rated) / len(rated)
    return round(abs(direction) * mean_tier, 3)


def decide(claim: Claim, signals: Signals, evidence: EvidencePack,
           llm: LLMStance | None, grounding: GroundingResult | None,
           cfg: dict[str, Any]) -> ClaimResult:
    scfg = cfg.get("scoring", {})
    weights = scfg.get("weights", {})
    policy = signals.policy
    never = [c.casefold() for c in scfg.get("never_auto_verdict", [])]
    conflict_threshold = float(scfg.get("conflict_threshold", 0.40))

    direction, conflict, detail = evidence_direction(evidence, cfg)
    ret_strength = retrieval_strength(evidence, policy)
    enforced = grounding.enforced_stance if grounding else Stance.NOT_IN_CONTEXT

    breakdown = {
        "rule_score": signals.rule_score,
        "ml_prior": signals.ml_score,
        "retrieval_strength": ret_strength,
        "evidence_direction": direction,
        "evidence_conflict": conflict,
        "n_sources": float(evidence.n_unique),
        **detail,
    }

    def result(verdict: Verdict, gate: str, reasoning: str,
               confidence: float) -> ClaimResult:
        return ClaimResult(
            claim=claim, signals=signals, evidence=evidence, llm=llm,
            grounding=grounding, verdict=verdict,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            gate_fired=gate, reasoning=reasoning, score_breakdown=breakdown,
        )

    # --- GATE 1: policy escalation ------------------------------------- #
    if signals.category.casefold() in never or not policy.get("auto_verdict", True):
        return result(
            Verdict.ESCALATE, "gate_1_never_auto_verdict",
            f"Category '{signals.category}' is excluded from automated verdicts "
            f"by policy. Claims of this type carry the highest social harm, and "
            f"an incorrect automated label can cause the very harm the system "
            f"exists to prevent. Routed to human review.",
            0.0,
        )

    # --- GATE 2: not enough evidence ----------------------------------- #
    need = int(policy.get("evidence_min_sources", 2))
    if evidence.n_unique < need:
        detail_msg = (
            f" Retrieval errors: {'; '.join(evidence.errors)}."
            if evidence.errors else ""
        )
        return result(
            Verdict.INSUFFICIENT_EVIDENCE, "gate_2_min_sources",
            f"Found {evidence.n_unique} independent source(s); this category "
            f"requires {need}. No published fact-check was located in "
            f"{claim.lang_code} or English. This system does not guess."
            f"{detail_msg}",
            0.0,
        )

    # --- GATE 3: the model abstained ----------------------------------- #
    if llm is not None and enforced == Stance.NOT_IN_CONTEXT and grounding and grounding.passed:
        return result(
            Verdict.INSUFFICIENT_EVIDENCE, "gate_3_llm_abstained",
            "The retrieved passages do not address this claim. The model "
            "returned NOT_IN_CONTEXT, which is the correct answer when the "
            "evidence does not speak to the claim.",
            0.0,
        )

    # --- GATE 4: grounding validator rejected the output --------------- #
    if grounding is not None and not grounding.passed:
        return result(
            Verdict.INSUFFICIENT_EVIDENCE, "gate_4_grounding_rejected",
            "The model's output FAILED the grounding check and was discarded: "
            + " | ".join(grounding.reasons)
            + ". The span it quoted could not be verified against the source "
              "text, so the verdict was withheld rather than reported.",
            0.0,
        )

    # --- GATE 5: sources disagree -------------------------------------- #
    if conflict >= conflict_threshold:
        return result(
            Verdict.ESCALATE, "gate_5_source_conflict",
            f"Sources disagree (conflict={conflict:.2f} >= "
            f"{conflict_threshold:.2f}): refuting weight "
            f"{detail['refute_weight']}, supporting weight "
            f"{detail['support_weight']}. Routed to human review.",
            0.0,
        )

    # --- GATE 6: fuse --------------------------------------------------- #
    rate_conf = rating_confidence(evidence, direction, cfg)
    breakdown["rating_confidence"] = rate_conf

    if enforced == Stance.CONTRADICT:
        llm_signal, stance_basis = 1.0, "grounded model stance (CONTRADICT)"
        polarity = Verdict.REFUTED
    elif enforced == Stance.SUPPORT:
        llm_signal, stance_basis = 1.0, "grounded model stance (SUPPORT)"
        polarity = Verdict.SUPPORTED
    else:
        # No model stance available (LLM off, unreachable, or skipped). The
        # published ratings substitute for it - a human fact-checker's rating
        # is a verdict, not a guess.
        llm_signal = rate_conf
        stance_basis = "published ratings (no model stance available)"
        if direction < 0:
            polarity = Verdict.REFUTED
        elif direction > 0:
            polarity = Verdict.SUPPORTED
        else:
            return result(
                Verdict.INSUFFICIENT_EVIDENCE, "gate_6_no_direction",
                "Sources were retrieved but none carry a rating that clearly "
                "supports or refutes this claim.",
                0.0,
            )
    breakdown["stance_basis_is_model"] = 1.0 if enforced != Stance.NOT_IN_CONTEXT else 0.0

    score = (
        float(weights.get("llm", 0.45)) * llm_signal
        + float(weights.get("retrieval", 0.30)) * ret_strength
        + float(weights.get("ml", 0.15)) * signals.ml_score
        + float(weights.get("rule", 0.10)) * signals.rule_score
    )
    total_weight = (
        float(weights.get("llm", 0.45)) + float(weights.get("retrieval", 0.30))
        + float(weights.get("ml", 0.15)) + float(weights.get("rule", 0.10))
    )
    normalised = score / total_weight if total_weight else 0.0
    breakdown["fused_score"] = round(score, 3)
    breakdown["normalised_score"] = round(normalised, 3)

    threshold = float(policy.get("stance_threshold",
                                 scfg.get("stance_threshold", 0.70)))

    if normalised < threshold:
        return result(
            Verdict.INSUFFICIENT_EVIDENCE, "gate_6_below_threshold",
            f"Evidence points toward {polarity.value} but the fused confidence "
            f"is {normalised:.2f}, below this category's threshold of "
            f"{threshold:.2f}. Withheld rather than asserted.",
            normalised,
        )

    srcs = ", ".join(
        f"{p.publisher} ({p.rating})" for p in evidence.passages[:3] if p.rating
    ) or ", ".join(p.publisher for p in evidence.passages[:3])
    resurf = ""
    if evidence.any_resurfaced:
        gaps = [p.resurface_gap_days for p in evidence.passages
                if p.resurfaced and p.resurface_gap_days]
        if gaps:
            resurf = (
                f" RESURFACED CONTENT: this claim was first made about "
                f"{max(gaps)} days before it was reviewed - old debunked "
                f"material back in circulation."
            )
    llm_note = ""
    if llm and grounding and grounding.passed and llm.quoted_span:
        llm_note = (
            f' Model cited passage(s) {llm.cited_passage_ids}, quoting '
            f'"{llm.quoted_span[:120]}" - span verified verbatim against the '
            f"source."
        )

    return result(
        polarity, "gate_6_scored",
        f"{polarity.value} on {evidence.n_unique} source(s): {srcs}. "
        f"Stance basis: {stance_basis}.{llm_note}{resurf}",
        normalised,
    )
