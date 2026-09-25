"""
LANGUAGE LAYER — transliterate, then detect, then translate.

The order is the point. Indians routinely type Tamil and Hindi in Latin
script ("ithu unmaya?", "kya ye sach hai?"). A pipeline that runs language ID
on the raw string labels romanised Tamil as English, retrieves nothing, and
returns silent garbage. So romanised script is detected and converted FIRST.

Two backends, and which one ran is always recorded in the output:
  llm   - Ollama does transliteration + translation in one call. Better on
          colloquial text.
  rule  - indic_transliteration (ITRANS scheme). Deterministic, offline, and
          weaker on colloquial spellings. A labelled degradation, not a mock.
"""

from __future__ import annotations

import re
from typing import Any

from .models import LangResult

TAMIL = (0x0B80, 0x0BFF)
DEVANAGARI = (0x0900, 0x097F)

# Romanised function words. Used only to answer "is this Latin text actually
# an Indic language?" - never to decide meaning.
ROMANIZED_TA = {
    "illa", "irukku", "iruku", "panra", "pannunga", "panni", "enna", "ithu",
    "idhu", "adhu", "nalla", "romba", "vanakkam", "seri", "yen", "epdi",
    "sollu", "solli", "sollirukaaru", "solraaru", "naan", "nee", "avan",
    "aval", "enga", "unga", "saapda", "saapttaa", "poga", "varum", "aagum",
    "dhaan", "thaan", "mattum", "kandippa", "anuppunga", "ellarukkum",
    "unmaya", "unmai", "marundhu", "nu", "aagidum", "veliya", "paaru",
}
ROMANIZED_HI = {
    "hai", "hain", "nahi", "nahin", "kya", "yeh", "yah", "woh", "aur", "ki",
    "ka", "ke", "mein", "kar", "karo", "karein", "bhai", "accha", "bahut",
    "sab", "sabko", "log", "baat", "kaise", "kyun", "tha", "thi", "hoga",
    "raha", "rahi", "sach", "jhooth", "bhejo", "zaroor", "turant", "jaldi",
    "abhi", "dawa", "ilaj", "sarkar", "paisa",
}


def _count_scripts(text: str) -> dict[str, int]:
    counts = {"tamil": 0, "devanagari": 0, "latin": 0, "other": 0}
    for ch in text:
        cp = ord(ch)
        if not ch.isalpha():
            continue
        if TAMIL[0] <= cp <= TAMIL[1]:
            counts["tamil"] += 1
        elif DEVANAGARI[0] <= cp <= DEVANAGARI[1]:
            counts["devanagari"] += 1
        elif ch.isascii():
            counts["latin"] += 1
        else:
            counts["other"] += 1
    return counts


def _romanized_hits(text: str) -> tuple[int, int]:
    toks = set(re.findall(r"[a-z]+", text.lower()))
    return len(toks & ROMANIZED_TA), len(toks & ROMANIZED_HI)


def _rule_transliterate(text: str, lang_code: str) -> str | None:
    """ITRANS -> native script. Deterministic and offline."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        return None
    target = {"ta": sanscript.TAMIL, "hi": sanscript.DEVANAGARI}.get(lang_code)
    if target is None:
        return None
    try:
        return transliterate(text, sanscript.ITRANS, target)
    except Exception:  # noqa: BLE001
        return None


def process(text: str, cfg: dict[str, Any], llm=None) -> LangResult:
    lcfg = cfg.get("languages", {})
    supported = set(lcfg.get("supported", ["ta", "hi", "en"]))
    do_translit = bool(lcfg.get("transliterate_romanized", True))
    backend = lcfg.get("translit_backend", "auto")

    counts = _count_scripts(text)
    notes: list[str] = []

    # --- native-script cases are unambiguous --------------------------- #
    if counts["tamil"] > 0 and counts["tamil"] >= counts["devanagari"]:
        lang, romanized = "ta", False
    elif counts["devanagari"] > 0:
        lang, romanized = "hi", False
    else:
        # Latin script. Is it English, or an Indic language typed in Latin?
        ta_hits, hi_hits = _romanized_hits(text)
        if ta_hits >= 2 and ta_hits >= hi_hits:
            lang, romanized = "ta", True
            notes.append(f"romanised Tamil detected ({ta_hits} marker tokens)")
        elif hi_hits >= 2:
            lang, romanized = "hi", True
            notes.append(f"romanised Hindi detected ({hi_hits} marker tokens)")
        else:
            lang, romanized = "en", False

    if lang not in supported:
        notes.append(f"language '{lang}' not in supported set; treating as en")
        lang = "en"

    native_script = text
    pivot_en = text
    translit_method = "none"
    translate_method = "passthrough" if lang == "en" else "none"

    use_llm = (
        llm is not None
        and backend in ("auto", "llm")
        and llm.available()
    )

    if lang == "en":
        return LangResult(
            raw=text, native_script=text, lang_code="en", pivot_en=text,
            was_romanized=False, script_counts=counts,
            translit_method="none", translate_method="passthrough", notes=notes,
        )

    # --- LLM path: transliteration + translation in one call ----------- #
    if use_llm:
        native, english = llm.translate(text, lang)
        if native:
            native_script = native
            translit_method = "llm" if romanized else "llm(identity)"
        if english:
            pivot_en = english
            translate_method = "llm"
        if not english:
            notes.append("LLM translation failed; retrieval will use raw text")

    # --- rule fallback for romanised input ----------------------------- #
    if romanized and do_translit and translit_method in ("none", ""):
        ruled = _rule_transliterate(text, lang)
        if ruled and ruled != text:
            native_script = ruled
            translit_method = "rule(ITRANS)"
            notes.append(
                "rule-based ITRANS transliteration used - weaker on colloquial "
                "spellings than the LLM path"
            )

    if translate_method == "none":
        notes.append(
            "no English pivot available; English-path retrieval will reuse the "
            "original text"
        )

    return LangResult(
        raw=text,
        native_script=native_script,
        lang_code=lang,
        pivot_en=pivot_en,
        was_romanized=romanized,
        script_counts=counts,
        translit_method=translit_method,
        translate_method=translate_method,
        notes=notes,
    )
