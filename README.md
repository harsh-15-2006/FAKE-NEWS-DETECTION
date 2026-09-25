# 🧠 AI-Powered Fake News Detector

## Problem Statement ID

**PSID:** `AI_03`
**Problem Statement:** **AI-Powered Fake News Detector**

---

## 👥 Team Details

| Name                | Year     | Department                               | Institution                        |
| ------------------- | -------- | ---------------------------------------- | ---------------------------------- |
| **Arivuselvan S**   | III Year | Artificial Intelligence and Data Science | Coimbatore Institute of Technology |
| **Baladharunesh B** | III Year | Artificial Intelligence and Data Science | Coimbatore Institute of Technology |
| **Harshini A**      | III Year | Artificial Intelligence and Data Science | Coimbatore Institute of Technology |
| **Sarvesh A K**     | III Year | Artificial Intelligence and Data Science | Coimbatore Institute of Technology |

---

# 🚨 Solution Architecture

The **AI-Powered Fake News Detector** is a hybrid **Neuro-Symbolic AI system** designed to analyze news articles and determine whether the content is potentially fabricated, misleading, unverified, or credible.

The system combines:

* Rule-based deception detection
* Statistical Machine Learning
* Keyword and semantic retrieval
* Retrieval-Augmented Generation (RAG)
* Local LLM reasoning using Ollama
* Evidence-based verification
* Adaptive multi-factor scoring
* Tamil and Tanglish language handling

Unlike a system that relies only on an LLM or keyword matching, our architecture combines multiple independent signals and gives greater importance to retrieved authoritative evidence when available.

### Architecture Flow

```text
                    ┌──────────────────────────────┐
                    │       USER INPUT             │
                    │ Raw News Article / Text      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     1. PREPROCESSING         │
                    │                              │
                    │ • Text Cleaning              │
                    │ • Language Detection         │
                    │ • Claim Extraction           │
                    │ • Entity & Keyword Parsing   │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌──────────────────────┐       ┌──────────────────────┐
       │ 2. RULE-BASED        │       │ 3. ML CLASSIFIER     │
       │    SIGNALS            │       │                      │
       │                      │       │ • TF-IDF             │
       │ • Urgency            │       │ • N-Grams             │
       │ • Clickbait          │       │ • Logistic Regression │
       │ • Scam Patterns      │       │ • Deception Score     │
       │ • ALL CAPS           │       │                      │
       │ • !!! / ???          │       │                      │
       └──────────┬───────────┘       └──────────┬───────────┘
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                    ┌──────────────────────────────┐
                    │     4. SEARCH ROUTER         │
                    │                              │
                    │ Entity & Specificity Rules   │
                    └──────────────┬───────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
        │   KEYWORD   │    │   SEMANTIC  │    │   HYBRID    │
        │    BM25     │    │   Chroma    │    │ BM25 + Vec  │
        └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
               └──────────────────┼──────────────────┘
                                  ▼
                    ┌──────────────────────────────┐
                    │     5. RAG RETRIEVAL         │
                    │                              │
                    │ • LangChain WebBaseLoader    │
                    │ • Chroma Vector Search        │
                    │ • BM25 Retrieval              │
                    │ • Nomic Embeddings            │
                    │ • Authoritative Fact Sources  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │    6. OLLAMA / LOCAL LLM     │
                    │                              │
                    │ • Claim vs Evidence          │
                    │ • Support / Refutation       │
                    │ • Grounding Verification     │
                    │ • Anti-Hallucination Check   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      7. SCORE ENGINE          │
                    │                              │
                    │ ML + Rules + Evidence        │
                    │ Adaptive Multi-Factor Fusion │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       FINAL RESULT            │
                    │                              │
                    │ • Verdict                    │
                    │ • Confidence                  │
                    │ • Reasons                     │
                    │ • Red Flags                   │
                    │ • Authoritative Sources       │
                    │ • Bilingual Explanation       │
                    └──────────────────────────────┘
```

### Architecture Diagram

> Place the generated architecture image in the repository as:
>
> `docs/architecture.png`

```markdown
![AI-Powered Fake News Detector Architecture](docs/architecture.png)
```

---

# 🔄 Solution Workflow

### 1. Preprocessing

The submitted news article is first cleaned and normalized.

The preprocessing layer performs:

* URL and unnecessary punctuation removal
* Text normalization
* Language detection
* Claim extraction
* Entity extraction
* Keyword and acronym identification
* Date and mission-code extraction

The system supports:

* 🇬🇧 English
* 🇮🇳 Tamil
* 🔤 Tanglish / Romanized Tamil

Tamil text is identified using Unicode character analysis, while common Romanized Tamil expressions are used to identify Tanglish content.

---

### 2. Rule-Based Deception Detection

The rule-based layer identifies common linguistic and behavioral indicators associated with deceptive or scam-like content.

It analyzes:

* Urgency-based expressions
* Sensational headlines
* Clickbait phrases
* Financial scam patterns
* ALL-CAPS usage
* Excessive `!!!` and `???`
* Suspicious calls to action

The result is an explainable **Rule Score**.

---

### 3. Machine Learning Classification

The ML layer analyzes the linguistic characteristics of the article.

The pipeline uses:

* Sublinear TF-IDF
* Word and character n-grams
* Calibrated Logistic Regression
* Balanced class weights

The classifier produces a calibrated **deception probability**, which becomes one of the inputs to the final score engine.

---

### 4. Intelligent Search Router

Instead of always using the same retrieval strategy, the Search Router determines the appropriate retrieval method based on the characteristics of the query.

#### Keyword Search

Used when the article contains highly specific identifiers such as:

```text
GSLV F17
PSLV C62
Specific dates
Mission codes
Exact phrases
```

Uses **BM25** for high-precision lexical matching.

#### Semantic Search

Used for broad conceptual queries where exact keyword matching may not be sufficient.

Uses **ChromaDB** with dense vector embeddings.

#### Hybrid Search

Combines:

```text
BM25 + Vector Similarity
```

This allows the system to benefit from both exact keyword matching and semantic similarity.

---

### 5. RAG-Based Evidence Retrieval

The Retrieval-Augmented Generation layer retrieves supporting evidence from authoritative sources.

The system uses:

* **LangChain WebBaseLoader**
* **ChromaDB**
* **BM25Okapi**
* **Ollama `nomic-embed-text`**
* Authoritative fact databases and web sources

Retrieved documents are divided into smaller chunks before indexing.

The system maintains both:

```text
Dense Vector Index
        +
Sparse BM25 Index
        ↓
Hybrid Evidence Retrieval
```

Hybrid retrieval combines lexical precision with semantic recall.

---

### 6. Local LLM Reasoning

Retrieved evidence is passed to a locally running LLM through **Ollama**.

The LLM performs claim-versus-evidence reasoning and categorizes the claim as:

* **Contradicted**
* **Partially Misleading**
* **Supported**
* **Unverified**

The system also performs grounding checks against retrieved evidence to reduce incorrect LLM conclusions.

---

### 7. Adaptive Score Engine

The final decision is generated by combining three major signals:

```text
ML Score
    +
Rule Score
    +
Evidence Score
    ↓
Adaptive Score Engine
    ↓
Final Result
```

The architecture gives greater weight to authoritative evidence when direct fact-checking information is available.

When direct evidence is unavailable, the system relies more heavily on the statistical ML and rule-based signals.

---

# 🧩 Technology Stack

| Category                 | Technology                     | Purpose                                  |
| ------------------------ | ------------------------------ | ---------------------------------------- |
| **Programming Language** | Python 3.10+ / 3.13            | Core application and AI pipelines        |
| **AI Orchestration**     | LangChain                      | Document loading, chunking and retrieval |
| **Web Ingestion**        | LangChain WebBaseLoader        | Loading online documents                 |
| **Web Parsing**          | BeautifulSoup4                 | HTML/document processing                 |
| **Vector Database**      | ChromaDB                       | Persistent vector storage                |
| **Embeddings**           | Ollama `nomic-embed-text`      | Dense semantic embeddings                |
| **Lexical Retrieval**    | Rank-BM25                      | Keyword-based retrieval                  |
| **Hybrid Retrieval**     | RRF + Min-Max Blending         | Combining semantic and lexical retrieval |
| **Machine Learning**     | Scikit-Learn                   | Text classification                      |
| **Feature Extraction**   | TF-IDF                         | Text feature representation              |
| **Classifier**           | Calibrated Logistic Regression | Deception probability                    |
| **Local LLM**            | Ollama                         | Local claim-evidence reasoning           |
| **LLM Models**           | Llama 3.2 / Phi-3 / Mistral    | Local inference                          |
| **Language Detection**   | Langdetect + Custom Regex      | English/Tamil/Tanglish detection         |
| **Frontend**             | Streamlit                      | Interactive web dashboard                |
| **Model Persistence**    | Joblib                         | Saving and loading ML models             |
| **Version Control**      | Git & GitHub                   | Source control and collaboration         |

---

# 🧠 Key Innovations

### Neuro-Symbolic Verification

The system combines deterministic rules, statistical machine learning, information retrieval and LLM reasoning instead of depending on a single model.

### Tamil & Tanglish Support

The pipeline specifically handles:

```text
English
Tamil
Tanglish / Romanized Tamil
```

This enables fact-checking of regional and code-mixed content that may not follow formal English patterns.

### Hybrid Retrieval

The system combines:

```text
BM25 Lexical Search
        +
Chroma Semantic Search
        ↓
Hybrid Evidence
```

This allows exact entities and semantically related evidence to be retrieved together.

### Evidence-Grounded LLM Reasoning

The local LLM does not work solely from its pretrained knowledge. Retrieved evidence is provided as context for claim-versus-evidence analysis.

### Local & Privacy-Preserving AI

The embedding and LLM components can run locally through Ollama, allowing news text and analysis to remain on the local system.

### Explainable Results

Instead of returning only a classification, the system provides:

* Verdict
* Confidence
* Reasons
* Key indicators
* Retrieved evidence
* Source URLs
* Bilingual explanation

---

# 📊 Final Output

The application produces a structured result containing:

```text
┌──────────────────────────────────────┐
│          FINAL RESULT                │
├──────────────────────────────────────┤
│ Verdict                              │
│ Confidence                           │
│                                      │
│ Why?                                 │
│ • Evidence-based explanation         │
│                                      │
│ Key Indicators                       │
│ • Sensational cues                   │
│ • Suspicious patterns                │
│                                      │
│ Authoritative Sources                │
│ • Government / trusted sources       │
│                                      │
│ Bilingual Summary                    │
│ • English                            │
│ • தமிழ் விளக்கம்                    │
└──────────────────────────────────────┘
```

---

# 🌐 Evidence Sources

The RAG layer is designed to retrieve evidence from authoritative sources, including:

* **ISRO**
* **WHO**
* **PIB Fact Check**
* **RBI**
* **TNFCU**
* Other trusted official and verified sources

The architecture uses retrieved evidence and source metadata as part of the verification process.

---

# 🏗️ Project Architecture

```text
AI-Powered-Fake-News-Detector/
│
├── src/
│   ├── preprocessor.py
│   ├── language_module.py
│   ├── rule_signals.py
│   ├── ml_classifier.py
│   ├── llm_reasoner.py
│   └── score_engine.py
│
├── langchain_retrieval/
│   ├── ingestion.py
│   ├── retrievers.py
│   └── router.py
│
├── models/
│   └── trained_models/
│
├── chroma_db_evidence/
│
├── app/
│   └── streamlit_app.py
│
├── docs/
│   └── architecture.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🎯 Objective

The objective of the **AI-Powered Fake News Detector** is to provide an explainable and evidence-driven approach to news verification by combining **Machine Learning, Rule-Based Analysis, Hybrid RAG Retrieval, LangChain and Local LLM Reasoning**.

The system is particularly designed to handle **English, Tamil and Tanglish content**, while providing users with evidence, source references and understandable explanations rather than relying solely on a single AI-generated prediction.

---

## 👨‍💻 Team

**AI_03 — AI-Powered Fake News Detector**

**Coimbatore Institute of Technology**
**Department of Artificial Intelligence and Data Science**
**III Year**
