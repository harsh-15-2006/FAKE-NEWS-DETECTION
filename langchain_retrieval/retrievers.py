import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_retrieval.ingestion import ingest_corpus

# Default configurable weights
KEYWORD_WEIGHT = 0.5
SEMANTIC_WEIGHT = 0.5

class EvidenceRetriever:
    """
    Modular Retriever managing:
    - Semantic Search (Chroma Vector Store)
    - Keyword Search (BM25 Okapi)
    - Hybrid Search (Score Normalization + Weighted Blend)
    """
    def __init__(self, vector_store: Optional[Chroma] = None, documents: Optional[List[Document]] = None):
        self.vector_store = vector_store
        self.documents: List[Document] = documents or []
        self.bm25: Optional[BM25Okapi] = None
        self._initialize()

    def _initialize(self):
        if not self.vector_store or not self.documents:
            db, docs = ingest_corpus(force_reload=False)
            self.vector_store = db
            self.documents = docs

        if self.documents:
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenizes text into words and acronyms, preserving mission IDs and hyphens.
        """
        return [w.lower() for w in re.findall(r'[\w\-]+', text)]

    def semantic_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Semantic vector search via Chroma.
        Returns top-k LangChain Document objects with similarity score attached to metadata.
        """
        if not self.vector_store:
            return []

        try:
            docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            results = []
            for doc, raw_score in docs_with_scores:
                score = float(raw_score) if raw_score is not None else 0.75
                # Clamp score
                score = max(0.0, min(1.0, score))
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "score": round(score, 3), "search_type": "semantic"}
                )
                results.append(doc_copy)
            return results
        except Exception as e:
            print(f"[Retriever] Semantic search fallback: {e}")
            docs = self.vector_store.similarity_search(query, k=k)
            return [
                Document(
                    page_content=d.page_content,
                    metadata={**d.metadata, "score": 0.70, "search_type": "semantic"}
                )
                for d in docs
            ]

    def keyword_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Lexical search using BM25 Okapi.
        Particularly effective for mission names, acronyms, dates, and exact phrases.
        """
        if not self.bm25 or not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        bm25_scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]

        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        results = []
        for idx in top_indices:
            raw_score = float(bm25_scores[idx])
            if raw_score <= 0.0:
                continue
            normalized_score = min(1.0, raw_score / max_bm25)
            doc = self.documents[idx]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "score": round(normalized_score, 3), "search_type": "keyword"}
            )
            results.append(doc_copy)

        return results

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        keyword_weight: float = KEYWORD_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT
    ) -> List[Document]:
        """
        Hybrid search combining normalized BM25 keyword relevance and vector similarity.
        """
        # Fetch larger pool for rank blending
        kw_docs = self.keyword_search(query, k=k * 2)
        sem_docs = self.semantic_search(query, k=k * 2)

        # Map by content identifier
        combined: Dict[str, Dict[str, Any]] = {}

        # Process Keyword Results
        for rank, d in enumerate(kw_docs):
            doc_id = d.metadata.get("url", "") + "::" + str(d.metadata.get("chunk_id", rank))
            combined[doc_id] = {
                "document": d,
                "kw_score": d.metadata.get("score", 0.0),
                "sem_score": 0.0
            }

        # Process Semantic Results
        for rank, d in enumerate(sem_docs):
            doc_id = d.metadata.get("url", "") + "::" + str(d.metadata.get("chunk_id", rank))
            if doc_id in combined:
                combined[doc_id]["sem_score"] = d.metadata.get("score", 0.0)
            else:
                combined[doc_id] = {
                    "document": d,
                    "kw_score": 0.0,
                    "sem_score": d.metadata.get("score", 0.0)
                }

        # Calculate weighted fused score
        total_weight = keyword_weight + semantic_weight
        kw_w = keyword_weight / total_weight if total_weight > 0 else 0.5
        sem_w = semantic_weight / total_weight if total_weight > 0 else 0.5

        scored_results = []
        for entry in combined.values():
            fused_score = (kw_w * entry["kw_score"]) + (sem_w * entry["sem_score"])
            doc = entry["document"]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "score": round(fused_score, 3),
                    "search_type": "hybrid",
                    "bm25_score": entry["kw_score"],
                    "vector_score": entry["sem_score"]
                }
            )
            scored_results.append((fused_score, doc_copy))

        # Sort descending by fused score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_results[:k]]

# Singleton instance
_retriever_instance: Optional[EvidenceRetriever] = None

def get_retriever() -> EvidenceRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = EvidenceRetriever()
    return _retriever_instance

def semantic_search(query: str, k: int = 5) -> List[Document]:
    return get_retriever().semantic_search(query, k=k)

def keyword_search(query: str, k: int = 5) -> List[Document]:
    return get_retriever().keyword_search(query, k=k)

def hybrid_search(query: str, k: int = 5, keyword_weight: float = KEYWORD_WEIGHT, semantic_weight: float = SEMANTIC_WEIGHT) -> List[Document]:
    return get_retriever().hybrid_search(query, k=k, keyword_weight=keyword_weight, semantic_weight=semantic_weight)
