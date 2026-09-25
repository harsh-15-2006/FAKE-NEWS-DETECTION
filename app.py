"""
SATYA — Streamlit dashboard.

    streamlit run app.py

Shows every number the verdict was built from. Nothing is a black box: the
signal breakdown, the grounding check, the gate that fired, and the evidence
with links are all on screen.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import apply_overrides, load_config, resolve_api_key  # noqa: E402
from src.models import Verdict  # noqa: E402
from src.pipeline import analyze  # noqa: E402

st.set_page_config(page_title="SATYA — Claim Verification", layout="wide")

VERDICT_STYLE = {
    Verdict.SUPPORTED.value: ("#1b5e20", "SUPPORTED"),
    Verdict.REFUTED.value: ("#b71c1c", "REFUTED"),
    Verdict.INSUFFICIENT_EVIDENCE.value: ("#5d4037", "INSUFFICIENT EVIDENCE"),
    Verdict.ESCALATE.value: ("#4a148c", "ESCALATE — HUMAN REVIEW"),
}

EXAMPLES = {
    "— pick an example —": "",
    "Romanised Tamil (health)":
        "Indha marundhu saapttaa sugar level 3 naalla normal aagidum nu "
        "doctor sollirukaaru. Kandippa share pannunga!",
    "Romanised Hindi (political)":
        "Sarkar ne ghoshna ki hai ki har nagrik ko 50000 rupees milenge. "
        "Turant sabko bhejo!",
    "Tamil script (health)":
        "இந்த மருந்து சாப்பிட்டால் சர்க்கரை அளவு மூன்று நாளில் சரியாகிவிடும் "
        "என்று மருத்துவர் கூறுகிறார்.",
    "Hindi script (political)":
        "सरकार ने घोषणा की है कि हर नागरिक को नई योजना के तहत 50000 रुपये मिलेंगे।",
    "English (communal → always escalates)":
        "A temple was demolished last night and the police did nothing about it.",
    "English (unverifiable → refusal)":
        "A new metro line was approved for Erode this morning by the state cabinet.",
    "English article (multi-claim)":
        "The government announced a new scheme yesterday. Officials said 50000 "
        "people registered on the first day. A doctor claimed the accompanying "
        "health camp cured diabetes in three days. The cricket team won the "
        "match by 7 wickets.",
}


@st.cache_data(show_spinner=False)
def _base_config():
    return load_config()


def badge(verdict: str, confidence: float) -> str:
    colour, label = VERDICT_STYLE.get(verdict, ("#37474f", verdict))
    return (
        f'<div style="background:{colour};color:#fff;padding:10px 16px;'
        f'border-radius:6px;font-weight:700;font-size:1.05rem;'
        f'display:inline-block;">{label} &nbsp;·&nbsp; confidence '
        f'{confidence:.2f}</div>'
    )


# ------------------------------------------------------------------ sidebar #
st.sidebar.title("SATYA")
st.sidebar.caption("Claim-level verification · Tamil / Hindi / English")

cfg = dict(_base_config())

st.sidebar.subheader("Retrieval")
backend = st.sidebar.selectbox("Backend", ["hybrid", "api", "offline"], index=0)
dual_path = st.sidebar.checkbox("Dual-path (native + English)", value=True)
topk = st.sidebar.slider("Top-K passages", 1, 20, 8)
min_sources = st.sidebar.slider(
    "Min sources override (0 = use category policy)", 0, 5, 0
)
min_relevance = st.sidebar.slider("Min relevance (gate)", 0.0, 0.6, 0.18, 0.02)
st.sidebar.caption(
    "Retrieval always returns its top-K whether or not anything matched. "
    "Passages below this overlap with the claim are discarded before scoring. "
    "Set it to 0.0 to see what the system would do without the gate."
)
resurface_gap = st.sidebar.slider("Resurfacing gap (days)", 30, 730, 180, step=30)

st.sidebar.subheader("LLM (Ollama)")
llm_on = st.sidebar.checkbox("Enable LLM stage", value=True)
llm_model = st.sidebar.text_input("Model", value=cfg["llm"]["model"])
llm_only_ambiguous = st.sidebar.checkbox(
    "Skip LLM when a prior debunk already decides", value=True
)
llm_timeout = st.sidebar.slider("LLM timeout (s)", 15, 180, 90, step=15)

st.sidebar.subheader("Grounding validator")
grounding_on = st.sidebar.checkbox("Enable", value=True)
strictness = st.sidebar.radio("Strictness", ["normalized", "exact"], index=0)
min_span = st.sidebar.slider("Min quoted-span chars", 4, 40, 12)

st.sidebar.subheader("Fusion weights")
w_llm = st.sidebar.slider("LLM", 0.0, 1.0, 0.45, 0.05)
w_ret = st.sidebar.slider("Retrieval", 0.0, 1.0, 0.30, 0.05)
w_ml = st.sidebar.slider("ML prior", 0.0, 1.0, 0.15, 0.05)
w_rule = st.sidebar.slider("Rule lexicon", 0.0, 1.0, 0.10, 0.05)
st.sidebar.caption(
    "Evidence dominates by design. Rule + ML together are a tie-breaking "
    "prior, never the verdict — the evidence gates run before the score is "
    "consulted at all."
)

st.sidebar.subheader("Claims")
checkworthy = st.sidebar.slider("Check-worthiness threshold", 0.0, 1.0, 0.45, 0.05)

cfg = apply_overrides(cfg, {
    "retrieval.backend": backend,
    "retrieval.dual_path": dual_path,
    "retrieval.topk": topk,
    "retrieval.min_relevance": min_relevance,
    "retrieval.resurface_gap_days": resurface_gap,
    "llm.enabled": llm_on,
    "llm.model": llm_model,
    "llm.only_when_ambiguous": llm_only_ambiguous,
    "llm.timeout_s": llm_timeout,
    "grounding.enabled": grounding_on,
    "grounding.strictness": strictness,
    "grounding.min_span_chars": min_span,
    "scoring.weights.llm": w_llm,
    "scoring.weights.retrieval": w_ret,
    "scoring.weights.ml": w_ml,
    "scoring.weights.rule": w_rule,
    "claims.checkworthy_threshold": checkworthy,
})

if min_sources:
    cfg["_override_min_sources"] = min_sources

api_key = resolve_api_key(None)

# --------------------------------------------------------------------- main #
st.title("SATYA")
st.markdown(
    "**We do not ask whether text *looks* fake.** We extract the individual "
    "claims, retrieve real published evidence in the claim's own language and "
    "in English, and return a verdict with its sources — or state plainly that "
    "we cannot verify it."
)

if not api_key:
    st.warning(
        "No API key found. Live fact-check search is off; only the local "
        "corpus will be used. Save your key to `.apikey` in the project root."
    )

c1, c2 = st.columns([3, 1])
with c2:
    choice = st.selectbox("Examples", list(EXAMPLES.keys()))
with c1:
    text = st.text_area(
        "Message, claim, or article",
        value=EXAMPLES[choice],
        height=140,
        placeholder="Paste a WhatsApp forward, a claim, or a news paragraph…",
    )

run = st.button("Verify", type="primary", use_container_width=True)

if run and text.strip():
    log: list[str] = []
    with st.status("Running pipeline…", expanded=True) as status:
        def progress(msg: str) -> None:
            log.append(msg)
            status.write(msg)

        report = analyze(text, cfg, api_key, progress)
        status.update(label=f"Done in {report.aggregate['elapsed_s']}s",
                      state="complete", expanded=False)

    agg = report.aggregate
    lang = report.language

    if agg["synthetic_evidence_used"]:
        st.error(
            "**SYNTHETIC SAMPLE EVIDENCE IN USE.** Some passages below come "
            "from `cache/sample_corpus.json`, which is invented demo data with "
            "placeholder publishers and `example.invalid` URLs — not real "
            "fact-checks. Run `python tools/build_cache.py` with an API key to "
            "replace it with real ClaimReview records."
        )

    m = st.columns(5)
    m[0].metric("Claims", agg["n_claims"])
    m[1].metric("Language", f"{lang.lang_code}{' (rom)' if lang.was_romanized else ''}")
    m[2].metric("LLM calls", agg["llm_calls"])
    m[3].metric("Grounding rejections", agg["grounding_rejections"])
    m[4].metric("Elapsed", f"{agg['elapsed_s']}s")

    if not agg["llm_available"]:
        st.warning(
            "Ollama was not reachable — the LLM stage was skipped and verdicts "
            "fall back to published ratings. Start it with `ollama serve`."
        )

    with st.expander("Language layer", expanded=lang.was_romanized):
        lc = st.columns(2)
        lc[0].markdown(
            f"**Detected:** `{lang.lang_code}` &nbsp; "
            f"**Romanised:** `{lang.was_romanized}`\n\n"
            f"**Transliteration:** `{lang.translit_method}`\n\n"
            f"**Translation:** `{lang.translate_method}`\n\n"
            f"**Script counts:** `{lang.script_counts}`"
        )
        lc[1].markdown(
            f"**Native script**\n\n> {lang.native_script}\n\n"
            f"**English pivot**\n\n> {lang.pivot_en}"
        )
        for note in lang.notes:
            st.caption(f"note: {note}")

    st.divider()

    for r in report.claims:
        st.markdown(f"### Claim {r.claim.idx + 1}")
        st.markdown(f"> {r.claim.text_native}")
        if r.claim.text_en.strip() != r.claim.text_native.strip():
            st.caption(f"English: {r.claim.text_en}")

        st.markdown(badge(r.verdict.value, r.confidence), unsafe_allow_html=True)
        st.markdown(f"**Why:** {r.reasoning}")
        st.caption(f"gate fired: `{r.gate_fired}`")

        tabs = st.tabs(["Grounding check", "Signals", "Evidence", "Raw"])

        # ---- grounding -------------------------------------------------- #
        with tabs[0]:
            if r.grounding is None:
                st.info(
                    "The LLM stage did not run for this claim "
                    "(disabled, unavailable, or skipped because a rated prior "
                    "debunk already decided it)."
                )
            elif r.grounding.passed:
                st.success("**GROUNDING PASSED** — the quoted span was found "
                           "verbatim in the cited source.")
                for reason in r.grounding.reasons:
                    st.markdown(f"- {reason}")
                if r.llm and r.llm.quoted_span:
                    st.markdown(f"**Model quoted:** `{r.llm.quoted_span}`")
                    st.markdown(f"**Cited passages:** {r.llm.cited_passage_ids}")
            else:
                st.error("**GROUNDING REJECTED — model output discarded.**")
                for reason in r.grounding.reasons:
                    st.markdown(f"- {reason}")
                if r.llm:
                    st.markdown(f"**Model claimed stance:** `{r.llm.stance.value}`")
                    st.markdown(f"**Span it produced:** `{r.llm.quoted_span}`")
                st.info(
                    "The span could not be verified against the source text, so "
                    "the stance was forced to NOT_IN_CONTEXT and the verdict was "
                    "withheld. This is the safeguard working, not a bug."
                )

        # ---- signals ----------------------------------------------------- #
        with tabs[1]:
            s = r.signals
            sc = st.columns(4)
            sc[0].metric("Category", s.category, f"{s.category_confidence:.2f}")
            sc[1].metric("Rule score", f"{s.rule_score:.2f}", f"weight {w_rule}")
            sc[2].metric("ML prior", f"{s.ml_score:.2f}", f"weight {w_ml}")
            sc[3].metric("Retrieval",
                         f"{r.score_breakdown.get('retrieval_strength', 0):.2f}",
                         f"weight {w_ret}")
            st.caption(s.ml_basis)
            if s.rule_hits:
                st.markdown("**Lexicon / structural hits**")
                for h in s.rule_hits:
                    st.markdown(f"- `{h}`")
            else:
                st.caption("No manipulation-lexicon signals fired.")
            st.markdown("**Score breakdown**")
            st.json(r.score_breakdown)
            st.markdown("**Category policy applied**")
            st.json(s.policy)

        # ---- evidence ------------------------------------------------------ #
        with tabs[2]:
            ev = r.evidence
            st.caption(
                f"strategy `{ev.strategy_used}` · paths "
                f"`{', '.join(ev.paths_queried) or 'none'}` · "
                f"{ev.n_unique} relevant passage(s)"
            )
            if ev.n_dropped_low_relevance:
                st.info(
                    f"**Relevance gate discarded {ev.n_dropped_low_relevance} "
                    f"retrieved passage(s)** that were not about this claim "
                    f"(below {ev.min_relevance_applied}). Retrieval rank is not "
                    f"relevance — without this gate they would have counted as "
                    f"evidence."
                )
            for err in ev.errors:
                st.warning(err)
            if not ev.passages:
                st.info("No passages retrieved.")
            for p in ev.passages:
                flags = []
                if p.synthetic:
                    flags.append("SYNTHETIC SAMPLE")
                if p.resurfaced:
                    flags.append(f"RESURFACED +{p.resurface_gap_days}d")
                suffix = f"  ·  **{' · '.join(flags)}**" if flags else ""
                st.markdown(
                    f"**[{p.pid}] {p.publisher}** — rating: `{p.rating or 'n/a'}`"
                    f"  ·  relevance `{p.relevance:.2f}`"
                    f"  ·  lang `{p.language or '?'}`  ·  src `{p.source}`{suffix}"
                )
                st.markdown(f"> {p.text}")
                if p.url:
                    st.markdown(f"[{p.url}]({p.url})")
                st.markdown("---")

        # ---- raw ------------------------------------------------------------ #
        with tabs[3]:
            st.markdown("**Claim features (why this sentence was selected)**")
            st.json(r.claim.features)
            if r.llm:
                st.markdown(f"**Raw model output** (`{r.llm.model}`)")
                st.code(r.llm.raw or "(empty)", language="json")

        st.divider()

    st.subheader("Document summary")
    st.json(agg)

elif run:
    st.info("Enter some text first.")
