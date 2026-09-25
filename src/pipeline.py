"""
ORCHESTRATION — wires the layers together and manages the retrieval/LLM budget.

Flow (matches the architecture diagram exactly):

  INPUT -> PREPROCESS/LANGUAGE -> CLAIM EXTRACTION
        -> [RULE | ML PRIOR | CATEGORY]  ==> SIGNAL BUS ==\\
        -> SEARCH ROUTER -> RAG RETRIEVAL -> EVIDENCE PACK ||
        -> OLLAMA (grounded only) -> GROUNDING VALIDATOR   ||
        -> SCORE ENGINE  <=================================//
        -> FINAL RESULT -> DASHBOARD

The signal bus is the double line: rule score, ML prior and category are
computed once and consumed twice - by the router to pick a strategy, and by
the score engine to contribute to fusion. In the original hand drawing those
two signals reached the router and then vanished, which meant they never
influenced the verdict at all.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from . import claims as claims_mod
from . import grounding as grounding_mod
from . import language as language_mod
from . import router as router_mod
from . import rules as rules_mod
from . import score as score_mod
from .llm import OllamaClient
from .models import ClaimResult, Report, Signals, Verdict
from .retrieve import retrieve


STRONG_MATCH_RELEVANCE = 0.45


def _exact_debunk_present(evidence, policy) -> bool:
    """True when the corpus already contains a clearly-rated review that is a
    STRONG lexical match for this claim. Used by llm.only_when_ambiguous: if a
    prior debunk already decides it, spending CPU to re-derive that is waste.

    The relevance floor is essential. Skipping the LLM on a merely-rated but
    weakly-matching passage removes the one component that would have answered
    NOT_IN_CONTEXT, and the claim then gets a confident verdict from evidence
    that was never about it.
    """
    need = int(policy.get("evidence_min_sources", 2))
    rated = [
        p for p in evidence.passages
        if score_mod.rating_direction(p.rating) != 0
        and p.relevance >= STRONG_MATCH_RELEVANCE
    ]
    return len(rated) >= need


def analyze(text: str, cfg: dict[str, Any], api_key: str | None = None,
            progress: Callable[[str], None] | None = None) -> Report:
    t0 = time.time()

    def say(msg: str) -> None:
        if progress:
            progress(msg)

    llm_cfg = cfg.get("llm", {})
    llm = OllamaClient(cfg)
    llm_up = llm.available()
    if not llm_up and llm_cfg.get("enabled", True):
        say(f"LLM unavailable: {llm.last_error}")
    elif llm.last_error:
        say(f"LLM note: {llm.last_error}")

    # ---- L2/L3 language ------------------------------------------------ #
    say("Language layer: transliterate -> detect -> translate")
    lang = language_mod.process(text, cfg, llm if llm_up else None)
    say(
        f"  lang={lang.lang_code} romanised={lang.was_romanized} "
        f"translit={lang.translit_method} translate={lang.translate_method}"
    )

    # ---- L4 claims ------------------------------------------------------ #
    say("Claim extraction")
    extracted = claims_mod.extract(lang, cfg, llm if llm_up else None)
    say(f"  {len(extracted)} check-worthy claim(s)")

    max_llm_calls = int(llm_cfg.get("max_claims", 5))
    only_ambiguous = bool(llm_cfg.get("only_when_ambiguous", True))
    llm_calls = 0

    results: list[ClaimResult] = []

    for claim in extracted:
        say(f"Claim {claim.idx}: {claim.text_en[:70]}")

        # ---- signal bus (computed once, consumed twice) ---------------- #
        rule_score, rule_hits = rules_mod.score(
            claim.text_native, claim.lang_code, cfg
        )
        ml_score, ml_basis = rules_mod.ml_prior(
            claim.text_native, rule_score, rule_hits
        )
        category, cat_conf, policy, strategy, cat_hits = router_mod.route(
            claim.text_native, claim.text_en, claim.lang_code, cfg,
            cfg.get("_forced_strategy"),
        )
        policy["category_keyword_hits"] = cat_hits
        # An explicit CLI/UI override beats the category default, otherwise the
        # flag would silently do nothing because category policy always wins.
        override_min = cfg.get("_override_min_sources")
        if override_min:
            policy["evidence_min_sources"] = int(override_min)
            policy["evidence_min_sources_overridden"] = True
        signals = Signals(
            rule_score=rule_score, rule_hits=rule_hits,
            ml_score=ml_score, ml_basis=ml_basis,
            category=category, category_confidence=cat_conf, policy=policy,
        )
        say(f"  category={category} ({cat_conf:.2f}) strategy={strategy} "
            f"rule={rule_score:.2f}")

        # ---- L6 retrieval ---------------------------------------------- #
        evidence = retrieve(
            claim.text_native, claim.text_en, claim.lang_code,
            strategy, cfg, api_key,
        )
        say(f"  retrieved {evidence.n_unique} passage(s) via "
            f"{', '.join(evidence.paths_queried) or 'no path'}")
        for err in evidence.errors:
            say(f"  ! {err}")

        # ---- L7 LLM + grounding ---------------------------------------- #
        llm_out = None
        ground = None
        use_llm = (
            llm_up
            and policy.get("use_llm", True)
            and evidence.passages
            and llm_calls < max_llm_calls
        )
        if use_llm and only_ambiguous and _exact_debunk_present(evidence, policy):
            say("  LLM skipped: prior rated debunk already decides this "
                "(--llm-only-when-ambiguous)")
            use_llm = False

        if use_llm:
            say(f"  asking {llm.model} for stance over "
                f"{len(evidence.passages)} passage(s)")
            llm_out = llm.stance(claim.text_en or claim.text_native,
                                 evidence.passages)
            llm_calls += 1
            if llm_out is None:
                say(f"  ! LLM returned no parseable JSON ({llm.last_error})")
            else:
                ground = grounding_mod.validate(llm_out, evidence, cfg)
                if ground.passed:
                    say(f"  grounding PASSED -> {ground.enforced_stance.value}")
                else:
                    say("  grounding REJECTED the model output: "
                        + " | ".join(ground.reasons))

        # ---- L7 score engine -------------------------------------------- #
        res = score_mod.decide(claim, signals, evidence, llm_out, ground, cfg)
        say(f"  VERDICT {res.verdict.value} ({res.gate_fired})")
        results.append(res)

    counts: dict[str, int] = {v.value: 0 for v in Verdict}
    for r in results:
        counts[r.verdict.value] += 1

    aggregate = {
        "n_claims": len(results),
        "verdict_counts": counts,
        "any_resurfaced": any(r.evidence.any_resurfaced for r in results),
        "grounding_rejections": sum(
            1 for r in results if r.grounding and not r.grounding.passed
        ),
        "llm_calls": llm_calls,
        "llm_model": llm.model if llm_up else None,
        "llm_available": llm_up,
        "synthetic_evidence_used": any(
            p.synthetic for r in results for p in r.evidence.passages
        ),
        "api_key_present": bool(api_key),
        "elapsed_s": round(time.time() - t0, 2),
    }

    return Report(input_text=text, language=lang, claims=results,
                  aggregate=aggregate)
