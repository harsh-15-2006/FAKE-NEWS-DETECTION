"""
CATEGORY CLASSIFIER + SEARCH ROUTER.

Category is resolved before retrieval, not after, because category determines
WHERE the ground truth lives. A sports result is checkable against structured
data, a health claim against authority documents, a political claim against
the fact-check corpus, and a communal claim often cannot be auto-checked at
all. One classifier cannot serve all four - there is no single corpus to check
against.

The category also selects the strategy, the source set, the thresholds and
whether an automated verdict is permitted at all.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .config import category_policy

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def classify(text_native: str, text_en: str, lang_code: str,
             cfg: dict[str, Any]) -> tuple[str, float, dict[str, int]]:
    """Keyword-scored category assignment. Transparent and inspectable -
    every hit is reportable, which a black-box classifier cannot offer."""
    cats = cfg.get("_categories", {}).get("categories", {})
    haystack = _norm(f"{text_native} {text_en}")

    scores: dict[str, int] = {}
    for name, spec in cats.items():
        if name == "general":
            continue
        hits = 0
        for field in ("keywords_en", f"keywords_{lang_code}",
                      "keywords_romanized", "keywords_ta", "keywords_hi"):
            for kw in spec.get(field, []) or []:
                if kw and _norm(kw) in haystack:
                    hits += 1
        if hits:
            scores[name] = hits

    if not scores:
        return "general", 0.0, {}

    # SAFETY PRECEDENCE: a category whose policy forbids an automated verdict
    # wins on ANY hit, even against a higher-scoring category.
    #
    # Rationale, and this is deliberate asymmetry: a false positive here costs
    # one unnecessary human review. A false negative lets an automated verdict
    # be issued on communal content, which is the exact harm the escalation
    # gate exists to prevent. "A temple was demolished and the police did
    # nothing" hits both `religious` and `crime`; it must escalate.
    protected = [
        name for name in scores
        if cats.get(name, {}).get("auto_verdict") is False
    ]
    total = sum(scores.values())
    if protected:
        best = max(protected, key=lambda k: scores[k])
        return best, round(scores[best] / total, 3) if total else 0.0, scores

    best = max(scores, key=lambda k: scores[k])
    confidence = round(scores[best] / total, 3) if total else 0.0
    return best, confidence, scores


def select_strategy(category: str, claim_text: str, cfg: dict[str, Any],
                    forced: str | None = None) -> str:
    """
    keyword  - distinctive entities, numbers or quoted text (lexical match wins)
    semantic - paraphrased or colloquial rumour with no fixed wording
    hybrid   - default; reciprocal-rank fusion of both
    """
    if forced and forced != "auto":
        return forced

    policy = category_policy(cfg, category)
    preferred = policy.get("strategy", "hybrid")
    if preferred != "hybrid":
        return preferred

    has_number = bool(re.search(r"\d", claim_text))
    has_quote = '"' in claim_text or "'" in claim_text
    has_proper = bool(re.search(r"(?<!^)\b[A-Z][a-zA-Z]{2,}\b", claim_text))

    if (has_number or has_quote) and has_proper:
        return "keyword"
    if not has_number and not has_proper:
        return "semantic"
    return "hybrid"


def route(text_native: str, text_en: str, lang_code: str, cfg: dict[str, Any],
          forced_strategy: str | None = None
          ) -> tuple[str, float, dict[str, Any], str, dict[str, int]]:
    category, confidence, hits = classify(text_native, text_en, lang_code, cfg)
    policy = category_policy(cfg, category)
    strategy = select_strategy(category, text_en or text_native, cfg, forced_strategy)
    policy["strategy_selected"] = strategy
    return category, confidence, policy, strategy, hits
