from typing import Dict, Any, List

class ScoreEngine:
    def __init__(self, ml_weight: float = 0.35, rule_weight: float = 0.25, evidence_weight: float = 0.40):
        self.default_ml_weight = ml_weight
        self.default_rule_weight = rule_weight
        self.default_evidence_weight = evidence_weight

    def compute_final_score(
        self,
        ml_result: Dict[str, Any],
        rule_result: Dict[str, Any],
        reasoning_result: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        w_ml = custom_weights.get("ml_weight", self.default_ml_weight) if custom_weights else self.default_ml_weight
        w_rule = custom_weights.get("rule_weight", self.default_rule_weight) if custom_weights else self.default_rule_weight
        w_ev = custom_weights.get("evidence_weight", self.default_evidence_weight) if custom_weights else self.default_evidence_weight

        total_w = w_ml + w_rule + w_ev
        w_ml /= total_w
        w_rule /= total_w
        w_ev /= total_w

        s_ml = ml_result.get("fake_probability", 0.50)
        s_rule = rule_result.get("rule_score", 0.0)
        s_ev = reasoning_result.get("deception_score", 0.50)
        verdict_cat = reasoning_result.get("verdict_category", "Unverified")

        contra_count = retrieval_result.get("contradiction_count", 0)
        corrob_count = retrieval_result.get("corroboration_count", 0)
        num_evidence = retrieval_result.get("num_retrieved", 0)

        if contra_count > 0:
            s_ev = max(s_ev, 0.90)
            verdict_cat = "Contradicted"
        elif corrob_count > 0:
            s_ev = min(s_ev, 0.12)
            verdict_cat = "Supported"
        elif any(e.get("verdict") == "MISLEADING" for e in retrieval_result.get("evidence_items", [])):
            s_ev = max(s_ev, 0.75)
            verdict_cat = "Misleading"

        has_direct_fact_check = verdict_cat in ["Contradicted", "Supported", "Misleading"]

        if has_direct_fact_check and num_evidence > 0:
            w_ev = 0.55
            w_ml = 0.25
            w_rule = 0.20
        elif num_evidence == 0 or verdict_cat == "Unverified":
            w_ev = 0.15
            w_ml = 0.50
            w_rule = 0.35

        composite_score = (w_ml * s_ml) + (w_rule * s_rule) + (w_ev * s_ev)
        composite_score = max(0.0, min(1.0, composite_score))

        if composite_score >= 0.70:
            verdict = "Fabricated / Fake News"
            verdict_badge = "🚨 Fabricated / Fake News"
            confidence = composite_score
            verdict_color = "#ef4444"
        elif composite_score >= 0.50:
            verdict = "Potentially Misleading"
            verdict_badge = "⚠️ Potentially Misleading"
            confidence = composite_score
            verdict_color = "#f59e0b"
        elif composite_score >= 0.35:
            verdict = "Unverified / Inconclusive"
            verdict_badge = "❓ Unverified / Inconclusive"
            confidence = 0.58
            verdict_color = "#6b7280"
        else:
            verdict = "Likely Authentic / Credible"
            verdict_badge = "✅ Likely Authentic / Credible"
            confidence = 1.0 - composite_score
            verdict_color = "#10b981"

        confidence_pct = int(round(confidence * 100))

        why_explanations = []
        if reasoning_result.get("reasoning_points"):
            why_explanations.extend(reasoning_result["reasoning_points"])
        else:
            if s_ev > 0.7:
                why_explanations.append("Authoritative fact-checking databases contain contradictions against this claim.")
            elif s_ev < 0.3:
                why_explanations.append("Official sources and publications corroborate the claims made.")
            if s_rule > 0.4:
                why_explanations.append("Language exhibits sensationalism, artificial urgency, or viral forwarding markers.")
            if s_ml > 0.65:
                why_explanations.append("Statistical text patterns closely resemble historical misinformation patterns.")

        indicators = list(rule_result.get("indicators", []))
        if ml_result.get("top_features"):
            indicators.append(f"[ML TOKENS] Salient vocabulary features: {', '.join(ml_result['top_features'][:4])}")
        if retrieval_result.get("sources"):
            indicators.append(f"[SOURCES] Cross-referenced with: {', '.join(retrieval_result['sources'][:3])}")

        return {
            "verdict": verdict,
            "verdict_badge": verdict_badge,
            "confidence_pct": confidence_pct,
            "verdict_color": verdict_color,
            "composite_score": round(composite_score, 3),
            "score_breakdown": {
                "ml_probability": round(s_ml, 3),
                "ml_weight": round(w_ml, 2),
                "ml_contribution": round(w_ml * s_ml, 3),
                "rule_score": round(s_rule, 3),
                "rule_weight": round(w_rule, 2),
                "rule_contribution": round(w_rule * s_rule, 3),
                "evidence_score": round(s_ev, 3),
                "evidence_weight": round(w_ev, 2),
                "evidence_contribution": round(w_ev * s_ev, 3)
            },
            "why_reasoning": why_explanations,
            "indicators": indicators,
            "tamil_summary": reasoning_result.get("tamil_summary", ""),
            "retrieved_evidence": retrieval_result.get("evidence_items", []),
            "retrieved_sources": retrieval_result.get("sources", [])
        }
