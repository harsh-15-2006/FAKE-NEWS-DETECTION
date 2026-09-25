"""
RULE-BASED SIGNAL — multilingual manipulation lexicon + structural features.

This produces rule_score, which carries only 0.10 of the fusion weight. It is
a prior that breaks ties. It can never by itself produce a REFUTED verdict:
src/score.py runs the evidence gates before the score is consulted at all.

The lexicon is trilingual with a romanised variant per concept, because a
Tamil forward typed in Latin script scores 0.0 against an English-only
dictionary - not because it is clean, but because the dictionary cannot see it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_WS = re.compile(r"\s+")

_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x1F000, 0x1F2FF),
    (0xFE00, 0xFE0F),
)


def _norm(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _is_emoji(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)


def score(text: str, lang_code: str, cfg: dict[str, Any]) -> tuple[float, list[str]]:
    lex = cfg.get("_lexicon", {})
    concepts = lex.get("concepts", {})
    structural = lex.get("structural", {})

    low = _norm(text)
    total = 0.0
    hits: list[str] = []

    # --- lexicon concepts: native + romanised + english ---------------- #
    for name, spec in concepts.items():
        weight = float(spec.get("weight", 0.0))
        variants: list[str] = []
        variants += spec.get("english", []) or []
        variants += spec.get("romanized", []) or []
        variants += spec.get(f"native_{lang_code}", []) or []
        # Also check the other native lists - a message can be code-mixed.
        for other in ("ta", "hi"):
            if other != lang_code:
                variants += spec.get(f"native_{other}", []) or []

        for phrase in variants:
            if not phrase:
                continue
            if _norm(phrase) in low:
                total += weight
                hits.append(f"{name}:'{phrase}' (+{weight:.2f})")
                break  # one hit per concept; do not stack the same idea

    # --- structural signals, language-invariant ------------------------ #
    alpha = [c for c in text if c.isalpha()]
    if alpha:
        spec = structural.get("caps_ratio", {})
        ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if ratio >= float(spec.get("threshold", 0.30)):
            w = float(spec.get("weight", 0.0))
            total += w
            hits.append(f"caps_ratio:{ratio:.0%} (+{w:.2f})")

    spec = structural.get("exclamation_run", {})
    longest = max((len(m) for m in re.findall(r"!+", text)), default=0)
    if longest >= int(spec.get("threshold", 2)):
        w = float(spec.get("weight", 0.0))
        total += w
        hits.append(f"exclamation_run:{longest} (+{w:.2f})")

    spec = structural.get("emoji_density", {})
    if text:
        density = sum(1 for c in text if _is_emoji(c)) / len(text)
        if density >= float(spec.get("threshold", 0.04)):
            w = float(spec.get("weight", 0.0))
            total += w
            hits.append(f"emoji_density:{density:.1%} (+{w:.2f})")

    spec = structural.get("forwarded_marker", {})
    for pat in spec.get("patterns", []) or []:
        if _norm(pat) in low:
            w = float(spec.get("weight", 0.0))
            total += w
            hits.append(f"forwarded_marker:'{pat}' (+{w:.2f})")
            break

    spec = structural.get("unattributed_number", {})
    min_digits = int(spec.get("min_digits", 3))
    big_numbers = [n for n in re.findall(r"\d[\d,]*", text)
                   if len(n.replace(",", "")) >= min_digits]
    if big_numbers:
        # "Unattributed" = a large number with no capitalised source nearby.
        has_source = bool(re.search(r"(?<!^)\b[A-Z][a-zA-Z]{2,}\b", text))
        if not has_source:
            w = float(spec.get("weight", 0.0))
            total += w
            hits.append(f"unattributed_number:{big_numbers[0]} (+{w:.2f})")

    return min(1.0, round(total, 3)), hits


def ml_prior(text: str, rule_score: float, rule_hits: list[str]) -> tuple[float, str]:
    """
    Lightweight structural prior standing in for a trained classifier.

    STATED PLAINLY: this is NOT a trained model. Training a supervised veracity
    classifier is the approach this project argues against - on the LIAR
    dataset such models hit ~1.000 train and ~0.25 test, and rumour detectors
    lose up to 40% on chronological splits. We do not ship one.

    What this returns is a transparent density prior over the same structural
    features, carrying 0.15 of the fusion weight. It is labelled as a prior
    everywhere it surfaces, never as a classifier output.
    """
    words = len(text.split())
    if words < 5:
        # Too short for a density measure to mean anything: 1 signal over 1
        # word would otherwise read as a very high prior.
        return 0.0, (
            f"structural prior withheld: input is {words} word(s), "
            f"too short for a meaningful density measure"
        )
    density = len(rule_hits) / (words ** 0.5)
    prior = min(1.0, 0.6 * rule_score + 0.4 * min(1.0, density))
    basis = (
        f"structural prior (NOT a trained classifier): "
        f"{len(rule_hits)} signals over {words} words"
    )
    return round(prior, 3), basis
