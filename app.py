import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from config import (
    DEFAULT_OLLAMA_LLM,
    DEFAULT_SEARCH_MODE,
    DEFAULT_WEIGHTS,
    CORPUS_FILE,
    RAW_DOCS_DIR
)
from src.detector_pipeline import get_pipeline
from src.ingest import build_vector_store

# Streamlit Page Configuration
st.set_page_config(
    page_title="AI-Powered Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism, Modern Typography, Vibrant Accents)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        margin-bottom: 1.25rem;
    }
    
    .verdict-hero {
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        text-align: center;
        border: 2px solid;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-right: 6px;
    }
    
    .tag-tamil {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .tag-english {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    .tag-tanglish {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    
    .indicator-pill {
        background: rgba(239, 68, 68, 0.12);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.25);
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.9rem;
    }
    
    .evidence-card {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    
    .mono-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample News Presets for Quick Testing
SAMPLE_PRESETS = {
    "Select a preset sample...": "",
    "🚨 Fake: UNESCO Best National Anthem (English)": "BREAKING NEWS: UNESCO has officially declared the Indian National Anthem 'Jana Gana Mana' as the BEST in the world! Share this immediately with 10 friends and groups to make every citizen proud! DO NOT IGNORE!",
    "🚨 Fake: TNEB Power Disconnection Phishing (Tamil)": "அவசர செய்தி: உங்கள் மின் கட்டணம் செலுத்தப்படாததால் இன்று இரவு 9.30 மணிக்கு உங்கள் மின் இணைப்பு துண்டிக்கப்படும்! உடனடியாக இந்த எண்ணை தொடர்பு கொள்ளவும். மின்சார வாரிய அறிவிப்பு.",
    "🚨 Fake: Free Laptop Scheme (Tanglish)": "Iniku night kulla indha link click panni 10 friends ku WhatsApp la forward pannina Central Govt free laptop kedaikum. Don't miss this chance!",
    "🚨 Fake: Miracle Garlic Cure for Cancer & Viruses (English)": "SHOCKING TRUTH: Drinking concentrated boiled garlic water with lemon twice daily is a 100% guaranteed miracle cure that eliminates cancer tumors and viruses in 24 hours! Big pharma and doctors are hiding this secret from you!",
    "✅ Real: ISRO Lunar Discovery Confirmation (English)": "The Indian Space Research Organisation (ISRO) officially confirmed that the Laser-Induced Breakdown Spectroscopy (LIBS) instrument onboard the Chandrayaan Pragyan Rover has unambiguously detected Sulfur (S) on the lunar south polar surface.",
    "✅ Real: Pongal Gift Distribution Protocol (Tamil)": "தமிழ்நாடு முதலமைச்சர் உத்தரவின்படி அனைத்து குடும்ப அட்டைதாரர்களுக்கும் ரேஷன் கடைகள் வாயிலாக 1 கிலோ பச்சரிசி, 1 கிலோ சர்க்கரை மற்றும் முழுக் கரும்பு அடங்கிய பொங்கல் பரிசு தொகுப்பு விநியோகம் தொடங்கப்பட்டது."
}

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.markdown("### ⚙️ System Configuration")
    
    # Model Selection
    available_models = ["llama3.2:1b", "phi3:latest", "mistral:latest", "gemma3:1b"]
    selected_model = st.selectbox(
        "🧠 Ollama LLM Reasoning Model",
        options=available_models,
        index=0,
        help="Select local Ollama model for claim vs evidence reasoning."
    )
    
    # Search Router Mode
    router_options = {
        "Auto (Smart Router)": None,
        "Hybrid (BM25 + Chroma RRF)": "hybrid",
        "Semantic (Chroma Vector)": "semantic",
        "Keyword (BM25 Lexical)": "keyword"
    }
    selected_router_label = st.selectbox(
        "🔀 Search Router Retrieval Mode",
        options=list(router_options.keys()),
        index=0,
        help="Configure Stage 4 Search Router strategy."
    )
    search_override = router_options[selected_router_label]
    
    st.markdown("---")
    st.markdown("#### ⚖️ Stage 7 Scoring Weights")
    w_ml = st.slider("ML Classifier Weight", 0.0, 1.0, DEFAULT_WEIGHTS["ml_weight"], 0.05)
    w_rule = st.slider("Rule-Based Signals Weight", 0.0, 1.0, DEFAULT_WEIGHTS["rule_weight"], 0.05)
    w_ev = st.slider("RAG Evidence Weight", 0.0, 1.0, DEFAULT_WEIGHTS["evidence_weight"], 0.05)
    
    custom_weights = {
        "ml_weight": w_ml,
        "rule_weight": w_rule,
        "evidence_weight": w_ev
    }
    
    st.markdown("---")
    # Knowledge Base Stats
    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        st.markdown(f"**📚 Ground Truth Corpus:** `{len(corpus_data)}` fact checks")
        st.markdown(f"**🌐 Languages:** `English`, `தமிழ்`, `Tanglish`")
        st.markdown(f"**🏛️ Sources:** PIB, WHO, RBI, TNFCU, ICMR, ISRO")

# Main Interface Tabs
tab_detector, tab_kb, tab_arch = st.tabs([
    "🛡️ News Verification Engine",
    "📚 Fact-Check Knowledge Base",
    "📐 System Architecture"
])

# Initialize Pipeline
@st.cache_resource
def get_detector(model_name: str):
    return get_pipeline(model_name)

pipeline = get_detector(selected_model)
pipeline.set_ollama_model(selected_model)

# ----------------- TAB 1: DETECTOR -----------------
with tab_detector:
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.markdown('<div class="main-title">AI-Powered Fake News Detector</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Multi-stage neuro-symbolic verification with RAG, Machine Learning & Local Ollama Reasoning</div>', unsafe_allow_html=True)
    with col_head2:
        st.markdown("""
        <div style="text-align: right; padding-top: 10px;">
            <span class="metric-badge tag-tamil">தமிழ் Support</span>
            <span class="metric-badge tag-english">English</span>
            <span class="metric-badge tag-tanglish">Tanglish</span>
        </div>
        """, unsafe_allow_html=True)

    # Preset selection
    preset_choice = st.selectbox("⚡ Quick Test Presets (Click to load sample):", options=list(SAMPLE_PRESETS.keys()), index=0)
    initial_text = SAMPLE_PRESETS[preset_choice]

    # Article Input
    news_input = st.text_area(
        "Enter or paste the news article / headline to verify:",
        value=initial_text,
        height=140,
        placeholder="e.g. Paste a news article, viral WhatsApp forward, or breaking news claim in English, Tamil, or Tanglish..."
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1, 2.5])
    with col_btn1:
        analyze_btn = st.button("🔍 Verify & Analyze News", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)
        if clear_btn:
            st.rerun()

    if analyze_btn and news_input.strip():
        with st.spinner("Analyzing news through 7-stage detection pipeline..."):
            result = pipeline.analyze_news(
                news_text=news_input,
                search_mode_override=search_override,
                custom_weights=custom_weights
            )

        # Unpack Results
        s1 = result["stage1_preprocess"]
        s2 = result["stage2_rules"]
        s3 = result["stage3_ml"]
        s4 = result["stage4_routing"]
        s5 = result["stage5_retrieval"]
        s6 = result["stage6_reasoning"]
        s7 = result["stage7_final"]

        st.markdown("<br>", unsafe_allow_html=True)

        # ========================================================
        # FINAL RESULT HERO CARD (Step 7 Final Synthesis)
        # ========================================================
        verdict = s7["verdict"]
        badge = s7["verdict_badge"]
        confidence = s7["confidence_pct"]
        v_color = s7["verdict_color"]

        st.markdown(f"""
        <div class="verdict-hero" style="border-color: {v_color}; background: rgba({int(v_color[1:3], 16)}, {int(v_color[3:5], 16)}, {int(v_color[5:7], 16)}, 0.12);">
            <div style="font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em; color: {v_color}; font-weight: 700; margin-bottom: 6px;">
                VERIFICATION OUTCOME
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #ffffff; margin-bottom: 8px;">
                {badge}
            </div>
            <div style="font-size: 1.25rem; font-weight: 600; color: #cbd5e1;">
                Confidence Score: <span style="color: {v_color}; font-size: 1.4rem;">{confidence}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Core Verdict Columns: Why?, Indicators, Scores
        col_res1, col_res2 = st.columns([1.6, 1.2])

        with col_res1:
            st.markdown("### 💡 Why? (Factual Explanation)")
            for why in s7["why_reasoning"]:
                st.markdown(f"• **{why}**")

            if s7.get("tamil_summary"):
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin-top: 12px;">
                    <div style="font-weight: 700; color: #fbbf24; font-size: 0.9rem;">தமிழ் சுருக்கம் (Tamil Summary):</div>
                    <div style="color: #fef3c7; font-size: 0.95rem; margin-top: 4px;">{s7['tamil_summary']}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📑 Retrieved Authoritative Evidence & Sources")
            if s7["retrieved_evidence"]:
                for ev in s7["retrieved_evidence"]:
                    ev_verdict = ev.get("verdict", "N/A")
                    v_badge_color = "#ef4444" if ev_verdict in ["FALSE", "SCAM"] else ("#10b981" if ev_verdict == "TRUE" else "#f59e0b")
                    st.markdown(f"""
                    <div class="evidence-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-weight: 700; color: #38bdf8;">🏛️ {ev.get('source')}</span>
                            <span style="background: {v_badge_color}; color: #fff; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700;">{ev_verdict}</span>
                        </div>
                        <div style="font-weight: 600; font-size: 0.95rem; color: #f1f5f9; margin-bottom: 4px;">{ev.get('title')}</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; line-height: 1.4;">{ev.get('content')[:260]}...</div>
                        <div style="font-size: 0.75rem; color: #64748b; margin-top: 6px;">Relevance Score: {ev.get('relevance_score')} | Method: {ev.get('method')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No matching prior fact-check in database. Assessment performed via ML and Linguistic Signals.")

        with col_res2:
            st.markdown("### 🚩 Key Indicators & Signals")
            if s7["indicators"]:
                for ind in s7["indicators"]:
                    st.markdown(f'<div class="indicator-pill">{ind}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="indicator-pill" style="color: #86efac; border-color: #22c55e;">No sensationalist red flags detected.</div>', unsafe_allow_html=True)

            st.markdown("### 📊 Ensemble Score Breakdown")
            sb = s7["score_breakdown"]
            st.markdown(f"""
            <div class="glass-card mono-text" style="font-size: 0.85rem;">
                <div><b>1. ML Classifier:</b> prob={sb['ml_probability']} × wt={sb['ml_weight']} = <b>{sb['ml_contribution']}</b></div>
                <div style="margin-top: 6px;"><b>2. Rule Signals:</b> score={sb['rule_score']} × wt={sb['rule_weight']} = <b>{sb['rule_contribution']}</b></div>
                <div style="margin-top: 6px;"><b>3. RAG & LLM:</b> score={sb['evidence_score']} × wt={sb['evidence_weight']} = <b>{sb['evidence_contribution']}</b></div>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 8px 0;">
                <div><b>Composite Score:</b> <span style="font-size: 1.05rem; color: #38bdf8;">{s7['composite_score']}</span> / 1.00</div>
            </div>
            """, unsafe_allow_html=True)

        # ========================================================
        # INTERACTIVE 7-STAGE PIPELINE STEPPER
        # ========================================================
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Interactive 7-Stage Pipeline Visualizer")
        st.caption("Inspect inputs, intermediate embeddings, token weights, and reasoning for every stage of the proposed architecture.")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            with st.expander("1. Preprocess", expanded=False):
                st.markdown(f"**Detected Language:** `{s1['language_info']['label']}`")
                st.markdown(f"**Extracted Claims ({len(s1['extracted_claims'])}):**")
                for c in s1['extracted_claims']:
                    st.markdown(f"- {c}")
                st.markdown(f"**Entities:** `{', '.join(s1['entities']) if s1['entities'] else 'None'}`")
                st.markdown(f"**Amounts:** `{', '.join(s1['monetary_amounts']) if s1['monetary_amounts'] else 'None'}`")
                st.markdown(f"**Top Keywords:** `{', '.join(s1['top_keywords'][:6])}`")

        with col_s2:
            with st.expander("2. Rule Signals", expanded=False):
                st.markdown(f"**Rule Score:** `{s2['rule_score']} / 1.0`")
                st.markdown(f"**Risk Level:** `{s2['risk_level']}`")
                st.markdown(f"**ALL CAPS Ratio:** `{int(s2['caps_ratio']*100)}%`")
                st.markdown(f"**Exclamation Points:** `{s2['exclamation_count']}`")
                st.markdown(f"**Trigger Matches ({len(s2['detected_signals'])}):**")
                for sig in s2['detected_signals']:
                    st.markdown(f"- `{sig['category']}`: {sig['matches']}")

        with col_s3:
            with st.expander("3. ML Classifier", expanded=False):
                st.markdown(f"**ML Verdict:** `{s3['ml_label']}`")
                st.markdown(f"**Fake Probability:** `{s3['fake_probability']}`")
                st.markdown(f"**Real Probability:** `{s3['real_probability']}`")
                st.markdown(f"**Model Confidence:** `{int(s3['confidence']*100)}%`")
                st.markdown(f"**Salient TF-IDF Tokens:**")
                st.code(", ".join(s3['top_features']), language="text")

        with col_s4:
            with st.expander("4. Search Router", expanded=False):
                st.markdown(f"**Chosen Strategy:** `{s4['strategy']}`")
                st.markdown(f"**Keyword Weight (BM25):** `{s4['keyword_weight']}`")
                st.markdown(f"**Semantic Weight (Chroma):** `{s4['semantic_weight']}`")
                st.markdown(f"**Routing Reason:**")
                st.caption(s4['reasoning'])

        col_s5, col_s6, col_s7 = st.columns(3)
        with col_s5:
            with st.expander("5. RAG Retrieval", expanded=False):
                st.markdown(f"**Retrieval Mode:** `{s5['search_mode']}`")
                st.markdown(f"**Passages Retrieved:** `{s5['num_retrieved']}`")
                st.markdown(f"**Debunk Records Found:** `{s5['contradiction_count']}`")
                st.markdown(f"**Corroborating Records:** `{s5['corroboration_count']}`")
                st.markdown("**Retrieved Sources:**")
                for s in s5['sources']:
                    st.markdown(f"- {s}")

        with col_s6:
            with st.expander("6. Ollama Reasoning", expanded=False):
                st.markdown(f"**Engine Mode:** `{s6.get('mode', 'Ollama')}`")
                st.markdown(f"**Verdict Category:** `{s6['verdict_category']}`")
                st.markdown(f"**Deception Score:** `{s6['deception_score']}`")
                st.markdown("**LLM Comparative Reasoning:**")
                for r in s6['reasoning_points']:
                    st.markdown(f"- {r}")

        with col_s7:
            with st.expander("7. Score Engine", expanded=False):
                st.markdown(f"**Final Verdict:** `{s7['verdict']}`")
                st.markdown(f"**Confidence:** `{s7['confidence_pct']}%`")
                st.markdown(f"**Composite Score:** `{s7['composite_score']}`")
                st.json(s7['score_breakdown'])

# ----------------- TAB 2: KNOWLEDGE BASE -----------------
with tab_kb:
    st.markdown("### 📚 Ground Truth Fact-Check Knowledge Base")
    st.write("Browse existing verified fact checks or upload new official PDFs / text announcements (CIT RAG Ingestion).")

    # Document upload
    with st.expander("➕ Ingest New Fact-Check Document (PDF or TXT)"):
        uploaded_file = st.file_uploader("Upload an official fact check or government press release:", type=["pdf", "txt"])
        if uploaded_file is not None:
            save_path = RAW_DOCS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved to {save_path.name}! Click below to re-index into Chroma DB.")
            if st.button("🚀 Re-index Chroma Vector Store"):
                with st.spinner("Chunking and generating embeddings with nomic-embed-text..."):
                    build_vector_store()
                    st.success("Indexing complete! Vector store and BM25 index updated.")
                    st.rerun()

    if os.path.exists(CORPUS_FILE):
        with open(CORPUS_FILE, "r", encoding="utf-8") as f:
            corpus_items = json.load(f)

        filter_lang = st.selectbox("Filter by Language:", ["All", "en", "ta", "tanglish"])
        for item in corpus_items:
            if filter_lang != "All" and item.get("language") != filter_lang:
                continue
            with st.container():
                c1, c2 = st.columns([3.5, 1])
                with c1:
                    st.markdown(f"#### {item['title']}")
                    st.markdown(f"**Claim:** *\"{item['claim']}\"*")
                    st.markdown(f"**Official Reality:** {item['explanation']}")
                    st.caption(f"Authority Source: {item['source']} | ID: {item['id']}")
                with c2:
                    v = item['verdict']
                    color = "#ef4444" if v in ["FALSE", "SCAM"] else ("#10b981" if v == "TRUE" else "#f59e0b")
                    st.markdown(f"""
                    <div style="background: {color}; color: #fff; padding: 6px 12px; border-radius: 6px; text-align: center; font-weight: 700; font-size: 0.85rem; margin-top: 20px;">
                        {v}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("---")

# ----------------- TAB 3: ARCHITECTURE -----------------
with tab_arch:
    st.markdown("### 📐 Proposed AI-Powered Fake News Detector Architecture")
    st.code("""
                    USER INPUT
                  (News Article)
                       │
                       ▼
              ┌─────────────────┐
              │ 1. PREPROCESS   │
              │ Clean text      │
              │ Extract claims  │
              │ Keywords/entities│
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
 ┌──────────────────┐      ┌──────────────────┐
 │ 2. RULE-BASED    │      │ 3. ML CLASSIFIER │
 │ SIGNALS          │      │                  │
 │ urgent           │      │ TF-IDF           │
 │ shocking         │      │ Logistic Reg.    │
 │ sponsored        │      │ / SVM            │
 │ click here       │      │                  │
 │ ALL CAPS         │      │ Fake/Misleading  │
 └────────┬─────────┘      └────────┬─────────┘
          │                         │
          └────────────┬────────────┘
                       ▼
              ┌─────────────────┐
              │ 4. SEARCH ROUTER│
              └────────┬────────┘
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Keyword   Semantic   Hybrid
             │         │         │
             └─────────┼─────────┘
                       ▼
              ┌─────────────────┐
              │ 5. RAG RETRIEVAL│
              │                 │
              │ Evidence        │
              │ Sources         │
              │ Similar claims  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 6. OLLAMA / LLM │
              │                 │
              │ Claim vs        │
              │ Evidence        │
              │ reasoning       │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ 7. SCORE ENGINE │
              │                 │
              │ ML + Rules +    │
              │ Evidence        │
              └────────┬────────┘
                       ▼
          ┌─────────────────────────┐
          │ FINAL RESULT            │
          │                         │
          │ Potentially Misleading  │
          │ Confidence: 84%         │
          │                         │
          │ Why?                    │
          │ Evidence                │
          │ Indicators              │
          └─────────────────────────┘
    """, language="text")

    st.markdown("""
    #### 🌟 Key Innovations & References
    1. **NidhiAI Language Module Integration**:
       - Multi-lingual preprocessing supporting **English**, **Tamil (தமிழ் script)**, and **Tanglish (Romanized Tamil)**.
       - Phonetic dictionary mapping and normalizer adapted for news domain (`vadhanthi`, `unmai`, `athirchi`, `seithi`, `panam`, `udane`).
    2. **CIT RAG Architecture Integration**:
       - Chroma vector database persistent storage.
       - Ollama embeddings with `nomic-embed-text`.
       - Document chunking with `RecursiveCharacterTextSplitter`.
       - PDF and TXT document ingestion pipeline for fact-checking reports.
    3. **Hybrid Search Router (RRF)**:
       - Reciprocal Rank Fusion blending BM25 Okapi lexical matching with dense vector cosine similarity.
    4. **Local Ollama LLM Reasoning**:
       - Claim-vs-evidence cross-examination using local models (`llama3.2:1b`, `phi3:latest`, `mistral:latest`).
    5. **Multi-Factor Score Engine**:
       - Adaptive weighted fusion dynamically shifting between ML, Rule signals, and authoritative RAG facts.
    """)
