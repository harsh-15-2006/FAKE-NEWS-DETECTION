import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw_docs"
MODELS_DIR = BASE_DIR / "models"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
CHROMA_EVIDENCE_DIR = BASE_DIR / "chroma_db_evidence"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Corpus and Model Files
CORPUS_FILE = DATA_DIR / "verified_news_corpus.json"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "fake_news_classifier.joblib"

# Ollama Settings
DEFAULT_OLLAMA_LLM = "llama3.2:1b"  # Fast and lightweight, phi3 and mistral also supported
DEFAULT_OLLAMA_EMBED = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Scoring Weights
DEFAULT_WEIGHTS = {
    "ml_weight": 0.35,
    "rule_weight": 0.25,
    "evidence_weight": 0.40
}

# Search Settings
DEFAULT_SEARCH_MODE = "hybrid"  # "keyword", "semantic", "hybrid"
TOP_K_EVIDENCE = 3
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
