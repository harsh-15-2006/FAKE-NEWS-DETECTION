#!/usr/bin/env python
"""
GROUNDING VALIDATOR TEST — run this on stage.

    python tests/test_grounding.py

This is the demo that wins the viva. It feeds deliberately-fabricated model
outputs into the validator and shows each one being caught. No network, no
Ollama, no API key, runs in under a second.

The point being proved: "we prompted it carefully" is a hope. This is an
enforced guarantee — every span the model produces is checked against the
source text, and anything it invented is discarded.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.grounding import validate  # noqa: E402
from src.models import EvidencePack, LLMStance, Passage, Stance  # noqa: E402


def make_evidence() -> EvidencePack:
    passages = [
        Passage(
            pid=0,
            text=("Claim reviewed: A doctor has said this medicine will bring "
                  "sugar levels back to normal in 3 days. Rating by "
                  "SAMPLE-FactChecker-A: False. Review title: No medicine "
                  "normalises blood sugar in three days"),
            publisher="SAMPLE-FactChecker-A",
            url="https://example.invalid/sample/health-001",
            rating="False",
            claim_date="2023-04-11",
            review_date="2023-04-18",
            language="en",
            source="sample",
            synthetic=True,
        ),
        Passage(
            pid=1,
            text=("Claim reviewed: Drinking hot water with lemon every morning "
                  "prevents all viral infections. Rating by SAMPLE-FactChecker-C: "
                  "False. Review title: There is no evidence that lemon water "
                  "prevents viral infection"),
            publisher="SAMPLE-FactChecker-C",
            url="https://example.invalid/sample/health-003",
            rating="False",
            claim_date="2020-03-14",
            review_date="2020-03-21",
            language="en",
            source="sample",
            synthetic=True,
        ),
    ]
    return EvidencePack(
        passages=passages, strategy_used="hybrid",
        paths_queried=["offline:en"], n_unique=2,
        any_resurfaced=False, synthetic_only=True,
    )


CASES = [
    (
        "HONEST: span copied verbatim from the cited passage",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[0],
            quoted_span="No medicine normalises blood sugar in three days",
            reasoning="The review rates the claim False.",
            raw="", model="test",
        ),
        True,
    ),
    (
        "HALLUCINATION: span invented, never appears in any passage",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[0],
            quoted_span="The World Health Organization confirmed this is dangerous",
            reasoning="WHO says so.",
            raw="", model="test",
        ),
        False,
    ),
    (
        "PARAPHRASE: close to the source but not copied from it",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[0],
            quoted_span="no drug can fix blood sugar within three days",
            reasoning="Reworded from the review.",
            raw="", model="test",
        ),
        False,
    ),
    (
        "FABRICATED CITATION: cites passage 7, only 0 and 1 were supplied",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[7],
            quoted_span="No medicine normalises blood sugar in three days",
            reasoning="Passage 7 says so.",
            raw="", model="test",
        ),
        False,
    ),
    (
        "UNWARRANTED: a confident stance with zero citations",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[],
            quoted_span="No medicine normalises blood sugar in three days",
            reasoning="It is obviously false.",
            raw="", model="test",
        ),
        False,
    ),
    (
        "TOO SHORT: a span of a few characters proves nothing",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[0],
            quoted_span="False",
            reasoning="Rated False.",
            raw="", model="test",
        ),
        False,
    ),
    (
        "ABSTENTION: NOT_IN_CONTEXT needs no citation and is always allowed",
        LLMStance(
            stance=Stance.NOT_IN_CONTEXT,
            cited_passage_ids=[],
            quoted_span="",
            reasoning="The passages do not address this claim.",
            raw="", model="test",
        ),
        True,
    ),
    (
        "FORMATTING ONLY: punctuation and case differ, wording identical",
        LLMStance(
            stance=Stance.CONTRADICT,
            cited_passage_ids=[0],
            quoted_span="no medicine normalises blood sugar in THREE days.",
            reasoning="Same sentence, different formatting.",
            raw="", model="test",
        ),
        True,
    ),
]


def main() -> int:
    cfg = load_config()
    evidence = make_evidence()

    print("\n" + "=" * 72)
    print("GROUNDING VALIDATOR — adversarial test")
    print(f"strictness={cfg['grounding']['strictness']}  "
          f"min_span_chars={cfg['grounding']['min_span_chars']}")
    print("=" * 72)

    passed = failed = 0
    for title, llm_out, expect_pass in CASES:
        result = validate(llm_out, evidence, cfg)
        ok = result.passed == expect_pass
        mark = "OK  " if ok else "BUG!"
        print(f"\n[{mark}] {title}")
        print(f"        model said      : {llm_out.stance.value}")
        print(f"        validator        : "
              f"{'PASSED' if result.passed else 'REJECTED'}")
        print(f"        enforced stance  : {result.enforced_stance.value}")
        for reason in result.reasons:
            print(f"        - {reason}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 72)
    print(f"{passed} behaved as specified, {failed} did not")
    print("Rejected outputs are forced to NOT_IN_CONTEXT, which the score")
    print("engine turns into INSUFFICIENT_EVIDENCE at gate 4 — the system")
    print("withholds a verdict rather than reporting one it cannot verify.")
    print("=" * 72 + "\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
