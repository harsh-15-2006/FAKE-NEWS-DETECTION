# 🛡️ AI-Powered Fake News Detector

An advanced multi-stage neuro-symbolic Fake News & Misinformation Detection System combining **Rule-Based Heuristics**, **Machine Learning Classification**, **RAG Retrieval (Chroma + BM25)**, and **Local Ollama LLM Reasoning**, with native **Tamil + English + Tanglish** multilingual support.

---

## 🏛️ Architectural Lineage & References

This project synthesizes and advances core concepts from two reference architectures:

1. **From `NidhiAI` (Multilingual Language Module)**:
   - Multilingual query handling supporting **English**, **Tamil (தமிழ்)**, and **Tanglish** (Romanized Tamil words like *iniku, vadhanthi, athirchi, seithi, unmai*).
   - Tamil script normalization, phonetic mapping, and entity extraction.
2. **From `CIT-AI-RAG-Assistant` (RAG Architecture)**:
   - Persistent **Chroma Vector Database** integration.
   - Dense vector embeddings via **Ollama `nomic-embed-text`**.
   - Recursive character chunking and PDF/TXT knowledge base document ingestion.

---

## 📐 System Architecture Diagram

```text
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
```

---

## 🔬 Pipeline Stage Breakdown

### 1. Preprocess (`src/preprocessor.py` & `src/language_module.py`)
- Cleans formatting artifacts, URLs, and excess whitespace.
- Detects whether input is English, Tamil script, or Tanglish code-mixed text.
- Applies phonetic dictionary normalizer (e.g. `vadhanthi -> vadhanthi`, `inniku -> iniki`).
- Extracts verifiable factual claims, numbers, currency amounts (`Rs`, `ரூ`), and uppercase acronyms (WHO, RBI, NASA, PIB, ISRO, TNEB).

### 2. Rule-Based Signals (`src/rule_signals.py`)
- Evaluates sensationalism cues across both English and Tamil:
  - **Urgency**: *"BREAKING", "SHARE IMMEDIATELY", "உடனே பகிருங்கள்", "அவசர செய்தி"*
  - **Sensationalism**: *"SHOCKING", "MIRACLE CURE", "அதிர்ச்சி தகவல்", "ரகசியம் அம்பலம்"*
  - **Scam / Sponsored**: *"Click here", "Earn daily", "இலவச லேப்டாப்", "மின் இணைப்பு துண்டிக்கப்படும்"*
  - **Formatting**: ALL CAPS proportion and repeated exclamation/question marks (`!!!`, `???`).

### 3. ML Classifier (`src/ml_classifier.py`)
- Character and word n-gram TF-IDF vectorizer trained on bilingual dataset.
- Calibrated probability model distinguishing authentic journalistic style from sensationalist fake text.
- Highlights top predictive TF-IDF tokens steering classification.

### 4. Search Router (`src/search_router.py`)
- Dynamically routes queries across:
  - **Keyword**: BM25 Okapi lexical matching (best for exact entity, phone numbers, acronyms).
  - **Semantic**: Chroma dense vector search with Ollama `nomic-embed-text`.
  - **Hybrid**: Reciprocal Rank Fusion (RRF) blending BM25 and Chroma vector search.

### 5. RAG Retrieval (`src/rag_retriever.py` & `src/ingest.py`)
- Pre-indexed ground truth database of verified fact-checks from official authorities:
  - **PIB Fact Check**, **WHO**, **RBI Clean Note Policy**, **Tamil Nadu Fact Check Unit (TNFCU)**, **ICMR**, **TANGEDCO**, **ISRO**.
- Dynamic ingestion capability: upload official PDFs or text bulletins to automatically re-index Chroma DB.

### 6. Ollama / LLM Reasoning (`src/llm_reasoner.py`)
- Prompts local Ollama model (`llama3.2:1b`, `phi3:latest`, or `mistral:latest`) to cross-examine extracted claims against retrieved evidence.
- Identifies direct contradictions, omission of context, or verified corroborations.
- Includes grounding verification to ensure strict factual fidelity.

### 7. Score Engine (`src/score_engine.py`)
- Adaptive multi-factor fusion:
  $$\text{Composite Score} = (w_{ml} \cdot S_{ml}) + (w_{rule} \cdot S_{rule}) + (w_{ev} \cdot S_{ev})$$
- Adaptively raises the weight of RAG evidence when authoritative ground truth debunking exists.
- Produces:
  - Final Verdict (`Fabricated / Fake News`, `Potentially Misleading`, `Unverified`, `Likely Authentic`)
  - Calibrated Confidence %
  - "Why?" explanation points
  - Red flag indicators
  - Retrieved sources and evidence citations
  - Bilingual summary (English + Tamil)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Ollama running locally (`ollama serve`) with models:
  ```bash
  ollama pull nomic-embed-text
  ollama pull llama3.2:1b
  ```

### Run the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
