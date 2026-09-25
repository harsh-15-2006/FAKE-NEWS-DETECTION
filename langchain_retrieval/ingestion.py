import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse

# Set User-Agent for web requests
os.environ["USER_AGENT"] = "NewsReliabilityAnalyzer/1.0 (Mozilla/5.0; AI Research Assistant)"

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db_evidence"
CACHE_FILE = PROJECT_ROOT / "data" / "evidence_corpus_cache.json"

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Specified Initial Retrieval Corpus URLs
URLS = [
    "https://www.indiatoday.in/science/story/india-divided-into-two-by-clouds-isro-satellite-captures-strange-view-from-space-3002522-2026-09-25",
    "https://www.isro.gov.in/Mission_GSLVF17.html",
    "http://isro.gov.in/Mission_PSLV_C62.html",
    "https://www.isro.gov.in/LVM3_M6_BlueBird_Block2_Mission.html",
    "https://www.who.int/news/item/24-09-2026-who-irch-working-group-2-advances-cooperation-on-herbal-medicine-quality-and-standards",
    "https://www.who.int/teams/who-global-traditional-medicine-centre/overview"
]

def classify_source(url: str) -> Tuple[str, str]:
    """
    Classifies the source and source_type based on URL.
    ISRO / WHO -> official
    India Today -> news
    """
    domain = urlparse(url).netloc.lower()
    if "isro.gov.in" in domain:
        return "ISRO", "official"
    elif "who.int" in domain:
        return "WHO", "official"
    elif "indiatoday.in" in domain:
        return "India Today", "news"
    else:
        # Extensible classification for future sources
        if any(tld in domain for tld in [".gov", ".org", ".edu"]):
            return domain, "official"
        return domain, "news"

def clean_page_content(text: str) -> str:
    """
    Cleans raw HTML text extracted from web pages: removes excessive linebreaks and script noise.
    """
    # Collapse multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def load_urls_with_fallback(urls: List[str]) -> List[Document]:
    """
    Loads web pages using LangChain WebBaseLoader with resilient URL normalization.
    """
    loaded_docs = []
    
    for url in urls:
        fetch_url = url
        # Resilient domain normalization (e.g. isro.gov.in without www fails DNS on some networks)
        if "isro.gov.in" in url and "www.isro.gov.in" not in url:
            fetch_url = url.replace("http://isro.gov.in", "https://www.isro.gov.in").replace("https://isro.gov.in", "https://www.isro.gov.in")
            
        source_name, source_type = classify_source(url)
        
        try:
            loader = WebBaseLoader(
                web_paths=[fetch_url],
                header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            docs = loader.load()
            for doc in docs:
                cleaned_text = clean_page_content(doc.page_content)
                if len(cleaned_text) > 100:
                    metadata = {
                        "source": source_name,
                        "source_type": source_type,
                        "url": url,
                        "title": doc.metadata.get("title", f"{source_name} Announcement")
                    }
                    loaded_docs.append(Document(page_content=cleaned_text, metadata=metadata))
            print(f"[Ingestion] Loaded: {url} ({source_name} - {source_type})")
        except Exception as e:
            print(f"[Ingestion] Warning fetching {url}: {e}")
            # Try original url if transformed, or vice versa
            try:
                loader = WebBaseLoader(web_paths=[url])
                docs = loader.load()
                for doc in docs:
                    metadata = {
                        "source": source_name,
                        "source_type": source_type,
                        "url": url,
                        "title": doc.metadata.get("title", f"{source_name} Announcement")
                    }
                    loaded_docs.append(Document(page_content=clean_page_content(doc.page_content), metadata=metadata))
            except Exception as e2:
                print(f"[Ingestion] Fallback failed for {url}: {e2}")

    return loaded_docs

def get_embedding_model():
    """
    Returns free/local embedding model (Ollama nomic-embed-text with local fallback).
    """
    try:
        embed = OllamaEmbeddings(model="nomic-embed-text")
        embed.embed_query("test probe")
        return embed
    except Exception as e:
        print(f"[Ingestion] Ollama embedding fallback active: {e}")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception:
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=768)

def ingest_corpus(force_reload: bool = False) -> Tuple[Chroma, List[Document]]:
    """
    Executes the ingestion pipeline:
    URL -> LangChain Web Loader -> Document -> Text splitting -> Embeddings -> Vector store
    Persists Chroma vector store locally to avoid re-generating embeddings.
    """
    embedding_model = get_embedding_model()
    
    # Check if vector store already exists and has documents
    if os.path.exists(CHROMA_DIR) and not force_reload:
        try:
            db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embedding_model)
            existing_data = db.get()
            if existing_data and "documents" in existing_data and len(existing_data["documents"]) > 0:
                print(f"[Ingestion] Reusing existing persisted vector store with {len(existing_data['documents'])} chunks.")
                # Reconstruct documents for BM25
                all_chunks = []
                for i, text in enumerate(existing_data["documents"]):
                    meta = existing_data["metadatas"][i] if "metadatas" in existing_data and existing_data["metadatas"] else {}
                    all_chunks.append(Document(page_content=text, metadata=meta))
                return db, all_chunks
        except Exception as e:
            print(f"[Ingestion] Re-indexing due to: {e}")

    print("[Ingestion] Loading corpus URLs via LangChain WebBaseLoader...")
    raw_documents = load_urls_with_fallback(URLS)

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", ";", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)

    # Attach chunk metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    print(f"[Ingestion] Created {len(chunks)} text chunks from {len(raw_documents)} documents.")

    # Create & Persist Chroma Vector Store
    os.makedirs(CHROMA_DIR, exist_ok=True)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"[Ingestion] Vector store persisted locally to {CHROMA_DIR}")

    return db, chunks

if __name__ == "__main__":
    ingest_corpus(force_reload=True)
