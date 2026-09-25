#  Claim-Level Misinformation Verification

**Hackathon:** Technical Hackathon, 4-Hour Event · **Domain:** AI / ML
**Problem statement:** AI-03 — AI-Powered Fake News Detection
**Languages:** Tamil · Hindi · English

---

## What it does

Most systems ask *"does this text look fake?"* That question has no good
answer. SATYA asks a different one: *"what exactly is this text claiming, and
has anyone already checked it?"* — and that question has an answer that comes
with a URL.

Given a WhatsApp forward, a claim, or a news article, SATYA:

1. detects romanised Tamil/Hindi and converts it to native script **before**
   language ID, then produces an English pivot;
2. splits the input into individual factual claims and scores each for
   check-worthiness;
3. routes each claim to a category, which decides the retrieval strategy, the
   source set, the thresholds, and whether an automated verdict is allowed;
4. retrieves published fact-checks along **two paths at once** — the claim's
   own language and the English pivot — and flags resurfaced content;
5. asks a local Ollama model for a stance **over the retrieved passages only**;
6. **verifies every span the model quotes against the source text** and
   discards anything it invented;
7. fuses the signals behind a set of evidence gates and returns one of four
   verdicts, always with its sources.

### The four verdicts

| | |
|---|---|
| `SUPPORTED` | evidence found and aligned; sources cited |
| `REFUTED` | contradicting evidence or a prior debunk; sources cited |
| `INSUFFICIENT_EVIDENCE` | retrieval came back thin, or the model abstained, or the grounding check rejected its output |
| `ESCALATE` | category policy forbids an automated verdict, or sources conflict |

`INSUFFICIENT_EVIDENCE` is a designed output, not a failure. It is the state a
trained classifier is architecturally incapable of producing.

---

## Quick start

```bash
pip install -r requirements.txt
python check_env.py                 # verifies Python, RAM, Ollama, packages, network

# the grounding validator, no network or Ollama needed
python tests/test_grounding.py

# one claim, offline
python cli.py --text "Indha marundhu saapttaa sugar level 3 naalla normal aagidum" \
              --backend offline --format text

# with live fact-check search (needs .apikey)
python cli.py --text "..." --backend hybrid --format text

# build the offline cache from real API records (do this while you have wifi)
python tools/build_cache.py

# dashboard
streamlit run app.py
```

**API key.** Save the Google Fact Check Tools API key to `.apikey` in the
project root, or set `GOOGLE_FACTCHECK_API_KEY`, or pass `--api-key`.

---

## Why not a trained classifier

This is the central design decision, and the evidence against the obvious
approach is unambiguous:

- **Shortcut learning.** On the LIAR dataset, XGBoost and Extra Trees reach
  training accuracy near 1.000 and collapse to ≈0.25 on test — a gap over 74%.
  The models memorise political entities and campaign slogans; those lexical
  shortcuts fail completely on unseen statements. The text alone contains
  insufficient veracity signals.
- **Collapse under realistic evaluation.** Rumour-detection models tested on
  chronological splits — simulating genuinely new rumours rather than random
  held-out data — drop by as much as 40%.
- **Style ≠ truth.** A detector can be highly accurate at identifying whether
  text was machine-generated while performing near chance on whether it is
  *true*.

So SATYA trains nothing. Retrieval-based verification does not have that
failure mode, because there is no decision boundary over topics to overfit.

---

## Architecture

```
USER INPUT  (text │ article │ image)
      │
      ▼
PREPROCESS / LANGUAGE          transliterate → detect → translate
      │                        (original text retained)
      ▼
CLAIM EXTRACTION               segment → check-worthiness → N claims
      │
      ├──────────────┬──────────────────┐        (per claim, independent)
      ▼              ▼                  ▼
 RULE-BASED     ML PRIOR          CATEGORY
 ta│hi│en       (structural)      CLASSIFIER
      └──────────────┴──────────────────┘
                     ▼
              ╔═════════════╗
              ║ SIGNAL BUS  ║═══════════════════╗
              ╚══════╤══════╝                   ║
                     ▼                          ║
              SEARCH ROUTER                     ║
        keyword │ semantic │ hybrid(RRF)        ║
                     ▼                          ║
              RAG RETRIEVAL                     ║
        dual path: ta/hi native + en pivot      ║
        union → dedupe → resurfacing check      ║
                     ▼                          ║
               EVIDENCE PACK                    ║
                     ▼                          ║
        OLLAMA — GROUNDED ONLY                  ║
        in: passages + claim, nothing else      ║
        out: JSON {stance, cited_ids,           ║
                   quoted_span, reasoning}      ║
                     ▼                          ║
        GROUNDING VALIDATOR                     ║
        span verbatim in a cited passage?       ║
        cited ids exist? stance warranted?      ║
        any failure → NOT_IN_CONTEXT            ║
                     ▼                          ║
        SCORE ENGINE  ◀═════════════════════════╝
        gates first, then weighted fusion
                     ▼
        SUPPORTED │ REFUTED │ INSUFFICIENT_EVIDENCE │ ESCALATE
                     ▼
                 DASHBOARD
```

### Three decisions that carry the design

**1. Transliteration runs before language ID.** Indians routinely type Tamil
and Hindi in Latin script. A pipeline that language-detects the raw string
labels romanised Tamil as English, retrieves nothing, and returns silent
garbage.

**2. The signal bus.** Rule score, ML prior and category are computed once and
consumed **twice** — by the router to pick a strategy, and by the score engine
to contribute to fusion. In the first draft of this architecture those two
signals reached the router and then vanished, which meant they never
influenced the verdict at all.

**3. The escalation gate lives in the score engine, not the router.** Policy is
applied at the point of decision, so a routing bug cannot bypass it.

---

## The grounding validator

This is the component that makes the LLM stage defensible.

A 3B model does not know Indian facts well enough to judge them. On the
IndicParam benchmark, 3–4B models score 16–50 on Indic factual tasks; even
Gemma4 31B reaches only 56.9–66.9% weighted accuracy across ten Indic
languages on L3Cube-IndicQuest v2. **So the model is never asked whether a
claim is true.** It receives the claim and the retrieved passages and nothing
else, and `NOT_IN_CONTEXT` is an explicitly legal answer.

That constraint is then *enforced*, not merely requested:

| Check | Catches |
|---|---|
| **Span check** | `quoted_span` must occur in a cited passage. Invented or paraphrased quotes die here. |
| **Citation check** | every cited passage id must exist in the evidence pack. Citing `[PASSAGE 7]` when only 0–2 were supplied is fabrication. |
| **Warrant check** | `SUPPORT`/`CONTRADICT` with zero citations is an unwarranted assertion. |

Any failure forces `NOT_IN_CONTEXT`, which the score engine turns into
`INSUFFICIENT_EVIDENCE` at gate 4. Every rejection is logged and shown in the
dashboard, because a caught hallucination is evidence the safeguard works.

`python tests/test_grounding.py` runs eight adversarial cases — invented spans,
paraphrases, fabricated citations, uncited assertions — and shows each being
caught. No network, no Ollama, under a second.

---

## The score engine

Gates run **before** the fused score is consulted, so a high rule score can
never on its own produce a `REFUTED` verdict:

```
1. category in never_auto_verdict      -> ESCALATE
2. n_sources < evidence_min_sources    -> INSUFFICIENT_EVIDENCE
3. LLM abstained (NOT_IN_CONTEXT)      -> INSUFFICIENT_EVIDENCE
4. grounding validator rejected output -> INSUFFICIENT_EVIDENCE
5. sources conflict above threshold    -> ESCALATE
6. otherwise                           -> fused score decides
```

```
final = 0.45·llm + 0.30·retrieval + 0.15·ml_prior + 0.10·rule
```

Evidence dominates by design. Rule and ML together are 25% — priors that break
ties, never the verdict.

---

## Category routing

Category decides four things at once, because different claim types have
ground truth in **different places**. A sports result is checkable against
structured data, a health claim against authority documents, a political claim
against the fact-check corpus, and a communal claim often cannot be
auto-checked at all.

| Category | Strategy | Harm | Auto-verdict |
|---|---|---|---|
| political | hybrid | 0.80 | yes, high threshold |
| health | hybrid | 0.95 | yes, strictest threshold |
| **religious / communal** | hybrid | 1.00 | **no — always escalates** |
| crime | hybrid | 0.85 | yes, with provenance |
| sports | keyword | 0.10 | yes, deterministic |
| entertainment | keyword | 0.15 | yes, low budget |
| finance / scam | hybrid | 0.85 | yes |

Communal content never receives an automated verdict. Religious
misinformation threatens social congruence and mob killings in India have
followed false rumours; an incorrect automated label there causes the very
harm the system exists to prevent. That is a hard-coded policy in
`config/categories.yaml`, enforced at gate 1.

Both published category distributions disagree on weights — ISB puts politics
at 46%, another study puts health at 27.2% and politics at 24.3%, and the mix
shifts by channel and period. That disagreement is exactly why this is a
config file and not a hardcoded prior.

---

## Honest limitations

Stated here so they are stated before anyone else finds them.

1. **Fact-check corpus coverage skews English.** Dual-path retrieval is the
   mitigation, not a cure. FactDRIL — the first large-scale multilingual
   fact-checking dataset for regional Indian languages — gathered 22,435
   samples from 11 IFCN-certified Indian sites over seven months: 9,058
   English, 5,155 Hindi, 8,222 across nine other languages.
2. **Translation quality.** With no `torch` in this build, Tamil/Hindi →
   English goes through the local Ollama model rather than IndicTrans2. The
   method used is recorded in every report (`translate_method`). IndicTrans2
   is the correct production choice and is on the roadmap.
3. **Semantic retrieval is token-overlap + BM25**, not embeddings, for the
   same reason. `rank_bm25` is pure Python; sentence-transformers is not.
4. **The "ML prior" is not a trained classifier** and is labelled as such
   everywhere it surfaces. Shipping a trained veracity classifier would
   contradict the argument this project is built on.
5. **`cache/sample_corpus.json` is synthetic** — invented records with
   placeholder publishers and `example.invalid` URLs, flagged
   `_synthetic: true`, halved in retrieval strength, and shown behind a red
   banner. It exists only so the pipeline is demonstrable with no network.
   Real evidence comes from `tools/build_cache.py`.

---

## Layout

```
FAKE-NEWS-DETECTION/
├── cli.py                     command line, every threshold a flag
├── app.py                     Streamlit dashboard
├── check_env.py               environment verification
├── requirements.txt
├── .apikey                    (you create this; not committed)
├── config/
│   ├── settings.yaml          all defaults
│   ├── categories.yaml        routing policy per category
│   └── lexicon.yaml           ta/hi/en manipulation lexicon
├── src/
│   ├── models.py              typed objects passed between layers
│   ├── config.py              config loading + CLI override merge
│   ├── language.py            transliterate → detect → translate
│   ├── claims.py              segmentation + check-worthiness
│   ├── rules.py               lexicon + structural signals
│   ├── router.py              category classifier + strategy selection
│   ├── retrieve.py            dual-path retrieval, RRF, resurfacing
│   ├── llm.py                 Ollama client, grounded prompting
│   ├── grounding.py           the validator
│   ├── score.py               gates + fusion
│   └── pipeline.py            orchestration
├── tests/test_grounding.py    adversarial validator test
├── tools/build_cache.py       fetch real ClaimReview records
└── cache/sample_corpus.json   labelled synthetic fallback
```

---

## Runtime

Python 3.10+ · **CPU only** · **zero models trained** · Ollama on
`localhost:11434` · works fully offline with `--backend offline`.

---

## Sources

**Why retrieval, not classification**
- Generalization Gaps in Political Fake News Detection (LIAR) — https://arxiv.org/html/2512.18533v1
- Generalizing Misinformation Detection to Unseen Events — https://lacuna.tiptreesystems.com/direction/generalizing-misinformation-detection-to-unseen-events/txn_c7cd5ed992f14bec8c09419aaa3bcfd5
- Towards Real-Time Fake News Detection under Evidence Scarcity — https://arxiv.org/html/2510.11277v1
- The Limitations of Stylometry for Detecting Machine-Generated Fake News — https://pith.science/paper/1908.09805

**Small models on Indic factual tasks**
- IndicParam benchmark — https://arxiv.org/pdf/2512.00333
- L3Cube-IndicQuest v2 — https://arxiv.org/pdf/2608.15535

**Retrieval**
- Google Fact Check Tools API — https://developers.google.com/fact-check/tools/api
- `claims.search` — https://developers.google.com/fact-check/tools/api/reference/rest/v1alpha1/claims/search

**Corpus and categories**
- FactDRIL — https://arxiv.org/pdf/2102.11276
- Fact-Checking India (ISB Institute of Data Science) — https://www.isb.edu/faculty-and-research/isb-institute-of-data-science/fact-checking-india
- Social Media Fake News in India (AJPOR) — https://www.ajpor.org/article/19049-social-media-fake-news-in-india

**Scale and impact**
- The Virality Gap (IJERT) — https://www.ijert.org/the-virality-gap-political-misinformation-and-the-information-crisis-in-india-s-digital-democracy-ijertv15is050041
- Images and misinformation in WhatsApp political groups (HKS Misinformation Review) — https://misinforeview.hks.harvard.edu/article/images-and-misinformation-in-political-groups-evidence-from-whatsapp-in-india/
- How India's Fact-Checking Ecosystem Works — https://www.indianrepublic.in/2026/06/how-indias-fact-checking-ecosystem-works.html
- PIB Fact Check Unit FAQ — https://www.pib.gov.in/FAQ_fact.aspx?reg=48&lang=2

**Regulatory driver**
- India's IT Amendment Rules 2026 — https://thelegalhubb.com/indias-it-amendment-rules-2026-explained-how-the-new-law-regulates-deepfakes-and-ai-generated-content/
- India targets deepfakes (Freshfields) — https://www.freshfields.com/en/our-thinking/blogs/technology-quotient/india-targets-deepfakes-and-ai-generated-content-key-changes-under-meitys-2026-102mjwn

**Language stack (roadmap targets)**
- AI4Bharat IndicTrans2 — https://github.com/AI4Bharat/IndicTrans2
- AI4Bharat models (IndicBERT, IndicNER, IndicXlit) — https://models.ai4bharat.org/
- Bhashini (MeitY) — https://www.microsoft.com/en-in/aifirstmovers/bhashini
