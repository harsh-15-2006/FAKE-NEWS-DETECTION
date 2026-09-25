"""
CLAIM EXTRACTION — segment, then score check-worthiness.

Rule-based and fully explainable, deliberately. Explainability is the product:
every claim carries the exact features that got it selected, so the dashboard
can show why this sentence was checked and that one was not.

A document is never given one label. Real articles mix true, false and
unverifiable claims - that is what makes them effective - so each claim is
carried forward independently.
"""

from __future__ import annotations

import re
from typing import Any

from .models import Claim, LangResult

# Sentence terminators across the three scripts, including the Devanagari danda.
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥\n])\s+|\n+")

_NUMBER = re.compile(r"\b\d[\d,.]*\b|[०-९]+|[௦-௯]+")
_DATE = re.compile(
    r"\b(19|20)\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b(today|yesterday|tomorrow|last week|last year|this week)\b",
    re.IGNORECASE,
)
_ASSERTIVE = re.compile(
    r"\b(is|are|was|were|has|have|had|will|did|does|said|says|announced|"
    r"confirmed|reported|found|proved|shows|showed|caused|banned|approved|"
    r"launched|died|killed|won|lost|increased|decreased)\b",
    re.IGNORECASE,
)
_HEDGE = re.compile(
    r"\b(may|might|could|possibly|perhaps|allegedly|reportedly|rumou?r|"
    r"i think|in my opinion|seems|apparently)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"\?\s*$")
# A capitalised token that is not sentence-initial, or any non-Latin word run.
_PROPER_LATIN = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-zA-Z]{2,}\b")
_NONLATIN_WORD = re.compile(r"[ऀ-ॿ஀-௿]{3,}")


def split_sentences(text: str, min_chars: int) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p and p.strip()]
    out: list[str] = []
    for p in parts:
        if len(p) >= min_chars:
            out.append(p)
        elif out:
            # Fold a fragment into the previous sentence rather than dropping it.
            out[-1] = f"{out[-1]} {p}".strip()
        else:
            out.append(p)
    return out


def score_checkworthy(sentence: str, cfg: dict[str, Any]) -> tuple[float, dict]:
    ccfg = cfg.get("claims", {})
    w = ccfg.get("weights", {})
    pen = ccfg.get("penalties", {})

    feats = {
        "has_number": bool(_NUMBER.search(sentence)),
        "has_date": bool(_DATE.search(sentence)),
        "has_proper_noun": bool(
            _PROPER_LATIN.search(sentence) or _NONLATIN_WORD.search(sentence)
        ),
        "has_assertive_verb": bool(_ASSERTIVE.search(sentence)),
        "adequate_length": 25 <= len(sentence) <= 400,
        "is_question": bool(_QUESTION.search(sentence)),
        "is_hedged": bool(_HEDGE.search(sentence)),
    }

    score = 0.0
    for key in ("has_number", "has_date", "has_proper_noun",
                "has_assertive_verb", "adequate_length"):
        if feats[key]:
            score += float(w.get(key, 0.0))
    if feats["is_question"]:
        score -= float(pen.get("is_question", 0.0))
    if feats["is_hedged"]:
        score -= float(pen.get("is_hedged", 0.0))

    return max(0.0, min(1.0, score)), feats


def extract(lang: LangResult, cfg: dict[str, Any], llm=None) -> list[Claim]:
    ccfg = cfg.get("claims", {})
    min_chars = int(ccfg.get("min_sentence_chars", 20))
    threshold = float(ccfg.get("checkworthy_threshold", 0.45))
    max_claims = int(ccfg.get("max_claims_per_doc", 12))

    native_sents = split_sentences(lang.native_script, min_chars)
    en_sents = split_sentences(lang.pivot_en, min_chars)

    claims: list[Claim] = []
    for i, s_native in enumerate(native_sents):
        # Pair by position; fall back to the whole pivot when counts differ,
        # which is normal because translation can merge or split sentences.
        s_en = en_sents[i] if i < len(en_sents) else lang.pivot_en
        score, feats = score_checkworthy(s_en if lang.lang_code != "en" else s_native, cfg)
        if score < threshold:
            continue
        claims.append(
            Claim(
                idx=len(claims),
                text_native=s_native,
                text_en=s_en,
                lang_code=lang.lang_code,
                checkworthy=round(score, 3),
                features=feats,
            )
        )
        if len(claims) >= max_claims:
            break

    # A short forward is one claim. Never return nothing for non-empty input.
    if not claims and lang.native_script.strip():
        score, feats = score_checkworthy(lang.pivot_en, cfg)
        claims.append(
            Claim(
                idx=0,
                text_native=lang.native_script.strip(),
                text_en=lang.pivot_en.strip(),
                lang_code=lang.lang_code,
                checkworthy=round(score, 3),
                features={**feats, "below_threshold_kept_as_whole_input": True},
            )
        )
    return claims
