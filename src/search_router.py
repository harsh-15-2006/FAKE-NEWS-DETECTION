import re
from typing import Dict, Any

class SearchRouter:
    def __init__(self, default_mode: str = "hybrid"):
        self.default_mode = default_mode

    def route_query(self, query: str, preprocess_info: Dict[str, Any] = None, manual_override: str = None) -> Dict[str, Any]:
        if manual_override and manual_override in ["keyword", "semantic", "hybrid"]:
            return {
                "selected_route": manual_override,
                "strategy": f"Manual Override: {manual_override.upper()}",
                "keyword_weight": 0.5 if manual_override == "hybrid" else (1.0 if manual_override == "keyword" else 0.0),
                "semantic_weight": 0.5 if manual_override == "hybrid" else (1.0 if manual_override == "semantic" else 0.0),
                "reasoning": f"User explicitly selected {manual_override.upper()} retrieval mode."
            }

        has_entities = False
        has_numbers = False
        is_short_query = len(query.split()) <= 4

        if preprocess_info:
            has_entities = bool(preprocess_info.get("entities")) or bool(preprocess_info.get("monetary_amounts"))
            has_numbers = bool(preprocess_info.get("dates_numbers"))
        else:
            has_entities = bool(re.findall(r'\b[A-Z]{2,6}\b', query))
            has_numbers = bool(re.search(r'\d+', query))

        if has_entities and has_numbers:
            selected = "hybrid"
            keyword_wt = 0.60
            semantic_wt = 0.40
            reason = "Detected high-specificity entities and numbers; combining lexical precision with semantic context."
        elif has_entities or has_numbers:
            selected = "hybrid"
            keyword_wt = 0.50
            semantic_wt = 0.50
            reason = "Detected specific entities/numerical claims; using balanced Hybrid search."
        elif is_short_query:
            selected = "semantic"
            keyword_wt = 0.20
            semantic_wt = 0.80
            reason = "Short conceptual query; semantic vector search provides better contextual recall."
        else:
            selected = "hybrid"
            keyword_wt = 0.40
            semantic_wt = 0.60
            reason = "Broad descriptive news claims; using Hybrid fusion (RRF) for optimal coverage."

        return {
            "selected_route": selected,
            "strategy": f"Auto-Routed: {selected.upper()}",
            "keyword_weight": keyword_wt,
            "semantic_weight": semantic_wt,
            "reasoning": reason,
            "has_entities": has_entities,
            "has_numbers": has_numbers
        }
