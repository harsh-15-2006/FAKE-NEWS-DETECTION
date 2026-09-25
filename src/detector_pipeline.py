import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
from src.preprocessor import preprocess_news
from src.rule_signals import analyze_rule_signals
from src.ml_classifier import get_ml_classifier
from src.search_router import SearchRouter
from src.rag_retriever import get_rag_retriever
from src.llm_reasoner import LLMReasoner
from src.score_engine import ScoreEngine
from config import DEFAULT_SEARCH_MODE, DEFAULT_OLLAMA_LLM

class FakeNewsDetectorPipeline:
    def __init__(self, ollama_model: str = DEFAULT_OLLAMA_LLM):
        self.ml_classifier = get_ml_classifier()
        self.search_router = SearchRouter(default_mode=DEFAULT_SEARCH_MODE)
        self.rag_retriever = get_rag_retriever()
        self.llm_reasoner = LLMReasoner(model_name=ollama_model)
        self.score_engine = ScoreEngine()

    def set_ollama_model(self, model_name: str):
        self.llm_reasoner.change_model(model_name)

    def analyze_news(
        self,
        news_text: str,
        search_mode_override: str = None,
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        stage1_preprocess = preprocess_news(news_text)
        cleaned_text = stage1_preprocess["cleaned_text"]
        normalized_text = stage1_preprocess["normalized_text"]
        claims = stage1_preprocess["extracted_claims"]
        lang_info = stage1_preprocess["language_info"]

        stage2_rules = analyze_rule_signals(news_text)
        stage3_ml = self.ml_classifier.predict(normalized_text)

        retrieval_query = " ".join(claims[:2]) if claims else cleaned_text[:200]
        stage4_routing = self.search_router.route_query(
            query=retrieval_query,
            preprocess_info=stage1_preprocess,
            manual_override=search_mode_override
        )
        selected_route = stage4_routing["selected_route"]
        kw_weight = stage4_routing["keyword_weight"]
        sem_weight = stage4_routing["semantic_weight"]

        stage5_retrieval = self.rag_retriever.retrieve_evidence(
            query=retrieval_query,
            search_mode=selected_route,
            keyword_weight=kw_weight,
            semantic_weight=sem_weight
        )

        stage6_reasoning = self.llm_reasoner.reason_claim_vs_evidence(
            news_text=cleaned_text,
            claims=claims,
            retrieved_context=stage5_retrieval["context_text"],
            evidence_items=stage5_retrieval["evidence_items"],
            language_info=lang_info
        )

        stage7_final = self.score_engine.compute_final_score(
            ml_result=stage3_ml,
            rule_result=stage2_rules,
            reasoning_result=stage6_reasoning,
            retrieval_result=stage5_retrieval,
            custom_weights=custom_weights
        )

        return {
            "stage1_preprocess": stage1_preprocess,
            "stage2_rules": stage2_rules,
            "stage3_ml": stage3_ml,
            "stage4_routing": stage4_routing,
            "stage5_retrieval": stage5_retrieval,
            "stage6_reasoning": stage6_reasoning,
            "stage7_final": stage7_final
        }

_pipeline_instance = None

def get_pipeline(model_name: str = DEFAULT_OLLAMA_LLM) -> FakeNewsDetectorPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FakeNewsDetectorPipeline(ollama_model=model_name)
    return _pipeline_instance
