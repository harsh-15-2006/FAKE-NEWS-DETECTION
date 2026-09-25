"""
Ollama client.

Two jobs only:
  1. translate  - Tamil/Hindi -> English (generation; quality is labelled)
  2. stance     - read supplied passages and say SUPPORT / CONTRADICT /
                  NOT_IN_CONTEXT (comprehension, NOT recall)

The model is NEVER asked "is this claim true?". It receives only the claim and
the retrieved passages, and NOT_IN_CONTEXT is an explicitly legal answer.
Anything it returns is then checked by src/grounding.py against the source
text; invented spans are discarded.

Why this matters, concretely: on the IndicParam benchmark, 3-4B models score
16-50 on Indic factual tasks. A 3B model does not know Indian facts well
enough to judge them. So it is never asked to.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from .models import LLMStance, Passage, Stance

STANCE_SYSTEM = """You are a strict evidence-reading tool. You do NOT know \
anything about the world. Your ONLY source of information is the numbered \
PASSAGES given to you.

Decide whether the PASSAGES support the CLAIM, contradict the CLAIM, or do \
not address it.

RULES - these are absolute:
1. Use ONLY the passages. Using your own knowledge is a FAILURE.
2. If the passages do not clearly address the claim, answer NOT_IN_CONTEXT. \
This is a correct and expected answer, not a failure.
3. quoted_span MUST be copied character-for-character from one of the \
passages you cite. Do not paraphrase it. Do not shorten words. If you cannot \
copy an exact span, answer NOT_IN_CONTEXT.
4. If stance is SUPPORT or CONTRADICT you MUST cite at least one passage id.

Reply with JSON only, exactly this shape:
{"stance":"SUPPORT|CONTRADICT|NOT_IN_CONTEXT","cited_passage_ids":[0],\
"quoted_span":"exact text copied from a cited passage","reasoning":"one short sentence"}"""

TRANSLATE_SYSTEM = """You are a transliteration and translation tool for Indian \
languages. Reply with JSON only, exactly this shape:
{"native_script":"the text written in its own native script",\
"english":"a plain, literal English translation"}

If the input is already in English, copy it into both fields.
Do not add commentary. Do not answer the content. Only convert it."""


class OllamaClient:
    def __init__(self, cfg: dict[str, Any]):
        lcfg = cfg.get("llm", {})
        self.base_url = lcfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.model = lcfg.get("model", "llama3.2:latest")
        self.fallback_model = lcfg.get("fallback_model")
        self.timeout = float(lcfg.get("timeout_s", 90))
        self.temperature = float(lcfg.get("temperature", 0.1))
        self.num_ctx = int(lcfg.get("num_ctx", 4096))
        self.json_mode = bool(lcfg.get("json_mode", True))
        self.enabled = bool(lcfg.get("enabled", True))
        self._available: bool | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ #

    def available(self) -> bool:
        if not self.enabled:
            return False
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            names = {m.get("name", "") for m in r.json().get("models", [])}
            self._available = True
            if self.model not in names and names:
                # Do not silently substitute - say so, then use what exists.
                self.last_error = (
                    f"model '{self.model}' not in ollama list {sorted(names)}; "
                    f"using '{sorted(names)[0]}'"
                )
                self.model = sorted(names)[0]
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"ollama unreachable: {exc}"
            self._available = False
        return self._available

    def _chat(self, system: str, user: str, model: str | None = None) -> str | None:
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.json_mode:
            payload["format"] = "json"
        try:
            r = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"ollama call failed: {exc}"
            return None

    @staticmethod
    def _parse_json(raw: str | None) -> dict[str, Any] | None:
        if not raw:
            return None
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Some models wrap JSON in prose or fences. Recover the outermost object.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    # ------------------------------------------------------------------ #

    def translate(self, text: str, lang_code: str) -> tuple[str | None, str | None]:
        """Returns (native_script, english). Either may be None on failure."""
        if not self.available():
            return None, None
        lang_name = {"ta": "Tamil", "hi": "Hindi", "en": "English"}.get(
            lang_code, lang_code
        )
        user = f"Language: {lang_name}\nInput text:\n{text}"
        data = self._parse_json(self._chat(TRANSLATE_SYSTEM, user))
        if not data:
            return None, None
        native = data.get("native_script")
        english = data.get("english")
        return (
            native if isinstance(native, str) and native.strip() else None,
            english if isinstance(english, str) and english.strip() else None,
        )

    def stance(self, claim_en: str, passages: list[Passage]) -> LLMStance | None:
        """Ask for a stance over the supplied passages. Never asks for truth."""
        if not self.available() or not passages:
            return None
        block = "\n\n".join(
            f"[PASSAGE {p.pid}] (publisher: {p.publisher}; rating: {p.rating})\n{p.text}"
            for p in passages
        )
        user = f"CLAIM:\n{claim_en}\n\nPASSAGES:\n{block}"
        raw = self._chat(STANCE_SYSTEM, user)
        data = self._parse_json(raw)
        if not data:
            return None

        stance_raw = str(data.get("stance", "")).strip().upper()
        try:
            stance = Stance(stance_raw)
        except ValueError:
            stance = Stance.NOT_IN_CONTEXT

        ids = data.get("cited_passage_ids", [])
        if isinstance(ids, int):
            ids = [ids]
        if not isinstance(ids, list):
            ids = []
        clean_ids: list[int] = []
        for v in ids:
            try:
                clean_ids.append(int(v))
            except (TypeError, ValueError):
                continue

        return LLMStance(
            stance=stance,
            cited_passage_ids=clean_ids,
            quoted_span=str(data.get("quoted_span", "") or ""),
            reasoning=str(data.get("reasoning", "") or ""),
            raw=raw or "",
            model=self.model,
        )
