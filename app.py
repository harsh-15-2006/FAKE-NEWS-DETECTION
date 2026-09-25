"""
FACTORA — dashboard.

    python -m streamlit run app.py

Plain-English interface. Every number the answer was built from is on screen.
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

st.set_page_config(page_title="FACTORA", page_icon="✅", layout="wide")

# Plain-English labels for the four possible answers.
VERDICT_UI = {
    Verdict.REFUTED.value: (
        "#b3261e", "✖", "FALSE",
        "Fact-checkers have already looked at this and found it wrong.",
    ),
    Verdict.SUPPORTED.value: (
        "#1b5e20", "✔", "TRUE",
        "Fact-checkers have looked at this and found it correct.",
    ),
    Verdict.INSUFFICIENT_EVIDENCE.value: (
        "#8a5a00", "?", "NOT ENOUGH PROOF",
        "Nobody has checked this yet, so we will not guess.",
    ),
    Verdict.ESCALATE.value: (
        "#5b3a9e", "!", "A PERSON MUST CHECK THIS",
        "This topic is too sensitive for a computer to judge on its own.",
    ),
}

EXAMPLES = {
    "— choose an example —": "",
    "Tamil typed in English letters (health)":
        "Indha marundhu saapttaa sugar level 3 naalla normal aagidum nu "
        "doctor sollirukaaru. Kandippa share pannunga!",
    "Hindi typed in English letters (politics)":
        "Sarkar ne ghoshna ki hai ki har nagrik ko 50000 rupees milenge. "
        "Turant sabko bhejo!",
    "Tamil script":
        "இந்த மருந்து சாப்பிட்டால் சர்க்கரை அளவு மூன்று நாளில் சரியாகிவிடும் "
        "என்று மருத்துவர் கூறுகிறார்.",
    "Hindi script":
        "सरकार ने घोषणा की है कि हर नागरिक को नई योजना के तहत 50000 रुपये मिलेंगे।",
    "A real false claim (should come back FALSE)":
        "Video shows PM Modi saying Today, India is the world's second "
        "largest beef producer",
    "A religious claim (a person must check it)":
        "A temple was demolished last night by a mob and the police did "
        "nothing about it.",
    "Something nobody has checked (should say NOT ENOUGH PROOF)":
        "The Erode city council approved a new metro line this morning.",
    "A short news story with several claims in it":
        "The government announced a new scheme yesterday. Officials said "
        "50000 people registered on the first day. A doctor claimed the "
        "health camp cured diabetes in three days. The cricket team won the "
        "match by 7 wickets.",
}


@st.cache_data(show_spinner=False)
def _base_config():
    return load_config()


def verdict_card(verdict: str, confidence: float) -> str:
    colour, icon, label, blurb = VERDICT_UI.get(
        verdict, ("#37474f", "", verdict, "")
    )
    return f"""
    <div style="background:{colour};color:#fff;padding:18px 22px;
                border-radius:10px;margin:6px 0 14px 0;">
      <div style="font-size:1.6rem;font-weight:800;letter-spacing:.5px;">
        {icon} &nbsp;{label}
      </div>
      <div style="font-size:1rem;opacity:.95;margin-top:6px;">{blurb}</div>
      <div style="font-size:.9rem;opacity:.85;margin-top:8px;">
        How sure we are: <b>{confidence:.0%}</b>
      </div>
    </div>"""


# =========================================================== SIDEBAR ===== #
st.sidebar.title("FACTORA")
st.sidebar.caption("Checks claims in Tamil, Hindi and English")

cfg = dict(_base_config())

st.sidebar.markdown("### Where we look for proof")
backend = st.sidebar.selectbox(
    "Search",
    ["hybrid", "api", "offline"],
    index=0,
    format_func=lambda v: {
        "hybrid": "Internet + saved copy (best)",
        "api": "Internet only",
        "offline": "Saved copy only (works with no internet)",
    }[v],
)
dual_path = st.sidebar.checkbox(
    "Search in the original language AND in English", value=True,
    help="A Tamil claim may have been checked in Tamil, in English, or both. "
         "We look in both places and combine what we find.",
)
topk = st.sidebar.slider("How many results to look at", 1, 20, 6)
min_relevance = st.sidebar.slider(
    "How closely a result must match", 0.0, 0.6, 0.18, 0.02,
    help="Search engines always return something, even when nothing really "
         "matches. Results below this are thrown away. Slide it to 0 to see "
         "what happens without this safety check.",
)
min_sources = st.sidebar.slider(
    "Minimum number of sources needed (0 = decide per topic)", 0, 5, 0
)

st.sidebar.markdown("### The AI reader (Ollama)")
llm_on = st.sidebar.checkbox("Use the AI reader", value=True)
llm_model = st.sidebar.text_input("Which model", value=cfg["llm"]["model"])
llm_num_gpu = st.sidebar.number_input(
    "Graphics card layers (0 = use the processor only)",
    min_value=0, max_value=99, value=0, step=1,
    help="Keep this at 0. Using the graphics card crashes on many laptops.",
)
llm_ctx = st.sidebar.select_slider(
    "How much text the AI reads at once", [1024, 2048, 4096], value=2048
)
llm_timeout = st.sidebar.slider("Give up after (seconds)", 15, 180, 90, step=15)
llm_only_ambiguous = st.sidebar.checkbox(
    "Skip the AI when fact-checkers already answered", value=True,
    help="Saves time. If a fact-checker has already rated this exact claim, "
         "we use their rating instead of asking the AI.",
)

st.sidebar.markdown("### Honesty check")
grounding_on = st.sidebar.checkbox("Check the AI for made-up quotes", value=True)
strictness = st.sidebar.radio(
    "How strict", ["normalized", "exact"], index=0,
    format_func=lambda v: {
        "normalized": "Ignore punctuation and capitals",
        "exact": "Character for character",
    }[v],
)
min_span = st.sidebar.slider("Shortest quote we accept (letters)", 4, 40, 12)

st.sidebar.markdown("### How much each thing counts")
w_llm = st.sidebar.slider("What the AI read", 0.0, 1.0, 0.45, 0.05)
w_ret = st.sidebar.slider("How much proof we found", 0.0, 1.0, 0.30, 0.05)
w_ml = st.sidebar.slider("Writing-style warning", 0.0, 1.0, 0.15, 0.05)
w_rule = st.sidebar.slider("Suspicious wording", 0.0, 1.0, 0.10, 0.05)
st.sidebar.caption(
    "Proof counts most. The two wording scores together are only a quarter, "
    "and they can never make something FALSE on their own."
)

cfg = apply_overrides(cfg, {
    "retrieval.backend": backend,
    "retrieval.dual_path": dual_path,
    "retrieval.topk": topk,
    "retrieval.min_relevance": min_relevance,
    "llm.enabled": llm_on,
    "llm.model": llm_model,
    "llm.num_gpu": int(llm_num_gpu),
    "llm.num_ctx": llm_ctx,
    "llm.timeout_s": llm_timeout,
    "llm.only_when_ambiguous": llm_only_ambiguous,
    "grounding.enabled": grounding_on,
    "grounding.strictness": strictness,
    "grounding.min_span_chars": min_span,
    "scoring.weights.llm": w_llm,
    "scoring.weights.retrieval": w_ret,
    "scoring.weights.ml": w_ml,
    "scoring.weights.rule": w_rule,
})
if min_sources:
    cfg["_override_min_sources"] = min_sources

api_key = resolve_api_key(None)

# ============================================================== MAIN ===== #
st.title("FACTORA")
st.markdown(
    "#### Paste a forwarded message. We find out if anyone has already "
    "checked it — and show you where."
)
st.markdown(
    "We do **not** guess whether a message *looks* fake. We pull out what it "
    "actually claims, search real fact-checking websites in the message's own "
    "language and in English, and show you the answer with a link. "
    "**If nobody has checked it, we say so instead of guessing.**"
)

if not api_key:
    st.warning(
        "No internet key set up, so we can only use the saved copy of "
        "fact-checks. Add your key to a file called `.apikey` in the project "
        "folder."
    )

c1, c2 = st.columns([3, 1])
with c2:
    choice = st.selectbox("Try an example", list(EXAMPLES.keys()))
with c1:
    text = st.text_area(
        "Message or news text",
        value=EXAMPLES[choice],
        height=150,
        placeholder="Paste a WhatsApp forward or a news paragraph here…",
    )

run = st.button("Check this", type="primary", use_container_width=True)

if run and text.strip():
    steps: list[str] = []
    with st.status("Working…", expanded=True) as status:
        def progress(msg: str) -> None:
            steps.append(msg)
            status.write(msg)

        report = analyze(text, cfg, api_key, progress)
        status.update(
            label=f"Finished in {report.aggregate['elapsed_s']} seconds",
            state="complete", expanded=False,
        )

    agg = report.aggregate
    lang = report.language
    lang_name = {"ta": "Tamil", "hi": "Hindi", "en": "English"}.get(
        lang.lang_code, lang.lang_code
    )

    if agg["synthetic_evidence_used"]:
        st.error(
            "**Some of the proof below is made-up demo data**, not real "
            "fact-checks. Run `python tools/build_cache.py` to replace it "
            "with real ones."
        )

    m = st.columns(4)
    m[0].metric("Things we checked", agg["n_claims"])
    m[1].metric(
        "Language", lang_name + (" (in English letters)" if lang.was_romanized else "")
    )
    m[2].metric("Made-up quotes caught", agg["grounding_rejections"])
    m[3].metric("Time taken", f"{agg['elapsed_s']}s")

    if not agg["llm_available"]:
        st.warning(
            "The AI reader was not running, so we used the ratings that "
            "fact-checkers published instead. The answers below are still "
            "backed by real sources."
        )

    if lang.was_romanized:
        with st.expander("We noticed this was typed in English letters", expanded=True):
            a, b = st.columns(2)
            a.markdown(
                f"**You typed** (in English letters)\n\n> {lang.raw}\n\n"
                f"**In {lang_name} letters**\n\n> {lang.native_script}"
            )
            b.markdown(f"**In English**\n\n> {lang.pivot_en}")
            st.caption(
                "This matters: if we had searched your text as-is, we would "
                "have looked for English words that do not exist and found "
                "nothing."
            )

    st.divider()

    for r in report.claims:
        st.markdown(f"### What it says")
        st.markdown(f"> {r.claim.text_native}")
        if r.claim.text_en.strip() != r.claim.text_native.strip():
            st.caption(f"In English: {r.claim.text_en}")

        st.markdown(verdict_card(r.verdict.value, r.confidence),
                    unsafe_allow_html=True)
        st.markdown(f"**Why:** {r.reasoning}")

        tabs = st.tabs([
            "Where we checked",
            "Did the AI make anything up?",
            "How we worked it out",
            "Technical details",
        ])

        # ---- evidence -------------------------------------------------- #
        with tabs[0]:
            ev = r.evidence
            if ev.n_dropped_low_relevance:
                st.info(
                    f"**We threw away {ev.n_dropped_low_relevance} search "
                    f"result(s)** because they were not actually about this "
                    f"claim. Search engines always return something — that "
                    f"does not make it proof."
                )
            if not ev.passages:
                st.warning("We found nothing that was really about this claim.")
            for p in ev.passages:
                notes = []
                if p.synthetic:
                    notes.append("DEMO DATA, NOT REAL")
                if p.resurfaced:
                    years = round((p.resurface_gap_days or 0) / 365, 1)
                    notes.append(
                        f"OLD STORY GOING AROUND AGAIN — first said about "
                        f"{years} years before it was checked"
                    )
                st.markdown(f"**{p.publisher}** said this is: **{p.rating or 'unrated'}**")
                if notes:
                    st.markdown(f":red[{' · '.join(notes)}]")
                st.caption(f"How closely it matches your text: {p.relevance:.0%}")
                st.markdown(f"> {p.text}")
                if p.url:
                    st.markdown(f"[Read the full fact-check]({p.url})")
                st.markdown("---")

        # ---- grounding ------------------------------------------------- #
        with tabs[1]:
            st.caption(
                "AI models sometimes invent quotes that sound right. Before we "
                "trust anything the AI says, we check that every quote it gives "
                "really appears in the source. If it does not, we throw the "
                "whole answer away."
            )
            if r.grounding is None:
                st.info(
                    "The AI was not used for this one — either it was turned "
                    "off, or fact-checkers had already answered it."
                )
            elif r.grounding.passed:
                st.success("**Checked and clean.** Every quote was found in the source.")
                if r.llm and r.llm.quoted_span:
                    st.markdown(f"The AI quoted: *“{r.llm.quoted_span}”*")
                    st.markdown("We found those exact words in the source. ✔")
            else:
                st.error("**The AI made something up. We threw its answer away.**")
                if r.llm:
                    st.markdown(f"It claimed the source said: *“{r.llm.quoted_span}”*")
                st.markdown("**What went wrong:**")
                for reason in r.grounding.reasons:
                    st.markdown(f"- {reason}")
                st.info(
                    "Because we could not find those words in the source, we "
                    "refused to give an answer instead of passing on something "
                    "invented. This is the safety check doing its job."
                )

        # ---- scoring ---------------------------------------------------- #
        with tabs[2]:
            s = r.signals
            topic = s.category.replace("_", " ").title()
            st.markdown(f"**Topic we think this is about:** {topic}")
            st.caption(
                "The topic decides where we look and how careful we are. "
                "Health and religious claims get the strictest treatment."
            )
            cols = st.columns(4)
            cols[0].metric("What the AI read", f"{w_llm:.0%}")
            cols[1].metric(
                "Proof found",
                f"{r.score_breakdown.get('retrieval_strength', 0):.0%}",
            )
            cols[2].metric("Writing style", f"{s.ml_score:.0%}")
            cols[3].metric("Suspicious wording", f"{s.rule_score:.0%}")

            if s.rule_hits:
                st.markdown("**Warning words we spotted in the message:**")
                for h in s.rule_hits:
                    st.markdown(f"- {h}")
            else:
                st.caption("No pushy or suspicious wording found.")
            st.caption(s.ml_basis)
            st.markdown(
                "Proof counts far more than wording. Suspicious wording alone "
                "can never make something FALSE."
            )

        # ---- raw --------------------------------------------------------- #
        with tabs[3]:
            st.markdown("**Why we picked this sentence to check**")
            st.json(r.claim.features)
            st.markdown("**All the numbers**")
            st.json(r.score_breakdown)
            st.markdown("**Rules applied for this topic**")
            st.json(r.signals.policy)
            if r.llm:
                st.markdown(f"**Exactly what the AI replied** ({r.llm.model})")
                st.code(r.llm.raw or "(nothing)", language="json")
            st.markdown("**Steps taken**")
            st.code("\n".join(steps))

        st.divider()

    st.subheader("Summary")
    counts = agg["verdict_counts"]
    sc = st.columns(4)
    sc[0].metric("False", counts.get("REFUTED", 0))
    sc[1].metric("True", counts.get("SUPPORTED", 0))
    sc[2].metric("Not enough proof", counts.get("INSUFFICIENT_EVIDENCE", 0))
    sc[3].metric("Needs a person", counts.get("ESCALATE", 0))
    with st.expander("Full technical summary"):
        st.json(agg)

elif run:
    st.info("Please paste some text first.")
