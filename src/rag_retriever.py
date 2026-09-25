import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Expose clean retrieval function implemented in the LangChain retrieval layer
from langchain_retrieval.rag_retriever import retrieve_evidence
from langchain_retrieval.retrievers import (
    semantic_search,
    keyword_search,
    hybrid_search,
    get_retriever
)
from langchain_retrieval.router import route_query
from langchain_retrieval.ingestion import ingest_corpus, URLS

class RAGRetriever:
    """Wrapper providing backward compatibility for detector_pipeline."""
    def __init__(self):
        self._retriever = get_retriever()

    def retrieve_evidence(self, query: str, search_mode: str = "hybrid", keyword_weight: float = 0.5, semantic_weight: float = 0.5, k: int = 3):
        # Calls the LangChain retrieval layer
        res = retrieve_evidence(query, k=k)
        evidence_items = res.get("evidence", [])
        sources = list(dict.fromkeys([e.get("source", "Unknown Authority") for e in evidence_items]))
        context_blocks = [f"[SOURCE: {e['source']} | TYPE: {e['source_type']}]\n{e['content']}" for e in evidence_items]
        context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No matching evidence found."
        
        return {
            "search_mode": res.get("strategy", search_mode),
            "evidence_items": evidence_items,
            "sources": sources,
            "context_text": context_text,
            "num_retrieved": len(evidence_items),
            "contradiction_count": 0,
            "corroboration_count": sum(1 for e in evidence_items if e.get("source_type") == "official")
        }

_rag_retriever_instance = None

def get_rag_retriever() -> RAGRetriever:
    global _rag_retriever_instance
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    return _rag_retriever_instance

__all__ = [
    "retrieve_evidence",
    "route_query",
    "semantic_search",
    "keyword_search",
    "hybrid_search",
    "get_retriever",
    "get_rag_retriever",
    "RAGRetriever",
    "ingest_corpus",
    "URLS"
]

