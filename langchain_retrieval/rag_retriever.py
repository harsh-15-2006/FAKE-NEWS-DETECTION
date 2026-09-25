import re
from typing import Dict, Any, List

# Reuse existing claim extraction from project preprocessor (avoiding duplicate implementation)
try:
    from src.preprocessor import extract_claims
except ImportError:
    # Graceful standalone fallback if called outside src root
    def extract_claims(text: str) -> List[str]:
        raw_sentences = re.split(r'(?<=[.!?\n])\s+', text)
        claims = [s.strip() for s in raw_sentences if len(s.strip()) > 20]
        return claims[:3] if claims else [text.strip()[:250]]

from langchain_retrieval.router import route_query
from langchain_retrieval.retrievers import (
    semantic_search,
    keyword_search,
    hybrid_search
)

def retrieve_evidence(article_text: str, k: int = 5) -> Dict[str, Any]:
    """
    Main LangChain Retrieval Entrypoint for News Reliability Analyzer.
    
    Pipeline:
    Article Text
         ↓
    Extract relevant claim/query
         ↓
    Search Router
         ↓
    ┌───────────┬────────────┬───────────┐
    │ Keyword   │ Semantic   │ Hybrid    │
    │ Search    │ Search     │ Search    │
    └───────────┴────────────┴───────────┘
                     ↓
            Retrieve relevant evidence
                     ↓
              Return documents and metadata

    Returns:
    {
        "claims": [...],
        "strategy": "hybrid",
        "router_reason": "...",
        "evidence": [
            {
                "content": "...",
                "source": "ISRO",
                "source_type": "official",
                "url": "...",
                "score": 0.91
            }
        ]
    }
    """
    if not article_text or not article_text.strip():
        return {
            "claims": [],
            "strategy": "semantic",
            "router_reason": "No article text provided.",
            "evidence": []
        }

    # 1. Extract searchable factual claim/query
    claims = extract_claims(article_text)
    primary_query = " ".join(claims[:2]) if claims else article_text.strip()[:300]

    # 2. Search Router (Keyword / Semantic / Hybrid)
    routing = route_query(primary_query)
    strategy = routing["strategy"]
    reason = routing["reason"]

    # 3. Selected Retriever
    if strategy == "keyword":
        docs = keyword_search(primary_query, k=k)
    elif strategy == "semantic":
        docs = semantic_search(primary_query, k=k)
    else:  # hybrid
        docs = hybrid_search(primary_query, k=k)

    # 4. Format evidence documents
    evidence_list = []
    for doc in docs:
        meta = doc.metadata or {}
        score = meta.get("score", 0.75)
        # Ensure score is rounded and non-negative
        score = round(max(0.0, min(1.0, float(score))), 2)

        evidence_list.append({
            "content": doc.page_content.strip(),
            "source": meta.get("source", "Unknown Authority"),
            "source_type": meta.get("source_type", "general"),
            "url": meta.get("url", ""),
            "score": score
        })

    return {
        "claims": claims if claims else [article_text.strip()[:200]],
        "strategy": strategy,
        "router_reason": reason,
        "evidence": evidence_list
    }

if __name__ == "__main__":
    # Test sample
    sample_text = "Did ISRO launch GSLV F17 for the upcoming communication mission?"
    result = retrieve_evidence(sample_text)
    print("Strategy:", result["strategy"])
    print("Reason:", result["router_reason"])
    print("Num evidence:", len(result["evidence"]))
