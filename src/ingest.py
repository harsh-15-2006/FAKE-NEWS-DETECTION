import os
import sys
import json
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config import CORPUS_FILE, RAW_DOCS_DIR, CHROMA_DB_DIR, DEFAULT_OLLAMA_EMBED, CHUNK_SIZE, CHUNK_OVERLAP

def load_fact_check_corpus() -> List[Document]:
    documents = []
    if not os.path.exists(CORPUS_FILE):
        return documents

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text_content = f"Title: {item['title']}\nClaim: {item['claim']}\nVerdict: {item['verdict']}\nExplanation: {item['explanation']}\nContent: {item.get('content', '')}"
        metadata = {
            "id": item["id"],
            "title": item["title"],
            "verdict": item["verdict"],
            "source": item["source"],
            "language": item.get("language", "en"),
            "claim": item["claim"]
        }
        documents.append(Document(page_content=text_content, metadata=metadata))

    return documents

def load_raw_documents() -> List[Document]:
    documents = []
    if not os.path.exists(RAW_DOCS_DIR):
        return documents

    for filename in os.listdir(RAW_DOCS_DIR):
        file_path = os.path.join(RAW_DOCS_DIR, filename)
        if filename.endswith(".pdf"):
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                for doc in pdf_docs:
                    doc.metadata["source"] = filename
                    doc.metadata["verdict"] = "OFFICIAL_DOCUMENT"
                documents.extend(pdf_docs)
            except Exception as e:
                print(f"[Ingest] Error loading PDF {filename}: {e}")
        elif filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": filename, "verdict": "OFFICIAL_DOCUMENT"}
                ))
            except Exception as e:
                print(f"[Ingest] Error loading TXT {filename}: {e}")

    return documents

def build_vector_store():
    all_docs = []
    all_docs.extend(load_fact_check_corpus())
    all_docs.extend(load_raw_documents())

    if not all_docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    try:
        embedding = OllamaEmbeddings(model=DEFAULT_OLLAMA_EMBED)
        embedding.embed_query("test")
    except Exception:
        from langchain_community.embeddings import FakeEmbeddings
        embedding = FakeEmbeddings(size=768)

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(CHROMA_DB_DIR)
    )

    return db

if __name__ == "__main__":
    build_vector_store()
