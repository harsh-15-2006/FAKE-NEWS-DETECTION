import re
import json
from typing import Dict, Any, List
from config import DEFAULT_OLLAMA_LLM, OLLAMA_BASE_URL

class LLMReasoner:
    def __init__(self, model_name: str = DEFAULT_OLLAMA_LLM):
        self.model_name = model_name
        self.llm = None
        self._init_llm()

    def _init_llm(self):
        try:
            from langchain_ollama import OllamaLLM
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=OLLAMA_BASE_URL,
                temperature=0.1,
                num_predict=180,
                timeout=20.0
            )
        except Exception as e:
            print(f"[LLM] Warning initializing Ollama LLM: {e}")
            self.llm = None

    def change_model(self, new_model_name: str):
        self.model_name = new_model_name
        self._init_llm()

    def reason_claim_vs_evidence(
        self,
        news_text: str,
        claims: List[str],
        retrieved_context: str,
        evidence_items: List[Dict[str, Any]],
        language_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        claims_str = "\n".join([f"- {c}" for c in claims[:4]]) if claims else news_text[:300]
        is_tamil = language_info.get("is_tamil", False) or language_info.get("is_tanglish", False)

        prompt = f"""
You are an expert AI Fact-Checking & Fake News Verification Analyst.
Analyze the following claim against the provided authoritative evidence context.

USER ARTICLE / CLAIMS:
{claims_str}

RETRIEVED FACT-CHECK EVIDENCE:
{retrieved_context}

INSTRUCTIONS:
1. Compare the claims directly with the retrieved evidence.
2. Determine if the claim is:
   - CONTRADICTED (Directly refuted by official fact-check or official authority)
   - PARTIALLY MISLEADING (Distorted context, exaggerated numbers, or phishing)
   - SUPPORTED / TRUE (Corroborated by reliable scientific or government sources)
   - UNVERIFIED (No direct authoritative evidence found, analyze for plausibility)
3. Explain WHY in 2-3 concise, bulleted factual points.
4. Give a deception severity score from 0.0 (Completely True) to 1.0 (Completely Fabricated).
{"5. Provide a short 1-line summary in Tamil (தமிழ் விளக்கம்) as well." if is_tamil else ""}

Format your answer as:
VERDICT_CATEGORY: [Contradicted | Misleading | Supported | Unverified]
DECEPTION_SCORE: [Number between 0.0 and 1.0]
REASONING:
- Point 1
- Point 2
TAMIL_SUMMARY: [Brief Tamil explanation or N/A]
"""

        if self.llm:
            try:
                raw_response = self.llm.invoke(prompt)
                parsed = self._parse_llm_response(raw_response, evidence_items, claims)
                parsed["mode"] = f"Ollama LLM ({self.model_name})"
                return parsed
            except Exception as e:
                print(f"[LLM] Ollama invocation fallback: {e}")

        return self._analytical_reasoning_fallback(claims, evidence_items, is_tamil)

    def _parse_llm_response(self, response_text: str, evidence_items: List[Dict[str, Any]], claims: List[str]) -> Dict[str, Any]:
        category = "Unverified"
        deception_score = 0.50
        reasoning_points = []
        tamil_summary = ""

        cat_match = re.search(r'VERDICT_CATEGORY:\s*([A-Za-z\s]+)', response_text, re.IGNORECASE)
        if cat_match:
            cat_str = cat_match.group(1).strip().lower()
            if "contradict" in cat_str:
                category = "Contradicted"
                deception_score = 0.90
            elif "mislead" in cat_str:
                category = "Misleading"
                deception_score = 0.75
            elif "support" in cat_str or "true" in cat_str:
                category = "Supported"
                deception_score = 0.10
            elif "unverif" in cat_str:
                category = "Unverified"
                deception_score = 0.50

        score_match = re.search(r'DECEPTION_SCORE:\s*([\d\.]+)', response_text, re.IGNORECASE)
        if score_match:
            try:
                deception_score = float(score_match.group(1))
            except ValueError:
                pass

        lines = [l.strip() for l in response_text.splitlines() if l.strip()]
        for line in lines:
            if line.startswith(('*', '-', '•')) or re.match(r'^\d+\.', line):
                clean_bullet = re.sub(r'^[*\-•\d\.\s]+', '', line).strip()
                if len(clean_bullet) > 12:
                    reasoning_points.append(clean_bullet)

        if not reasoning_points:
            if "REASONING:" in response_text:
                after_r = response_text.split("REASONING:", 1)[1]
                cand_lines = [l.strip() for l in after_r.splitlines() if l.strip()]
                for l in cand_lines:
                    if any(l.upper().startswith(p) for p in ["TAMIL_SUMMARY:", "DECEPTION_SCORE:"]):
                        break
                    c_clean = re.sub(r'^[*\-•\d\.\s]+', '', l).strip()
                    if len(c_clean) > 12:
                        reasoning_points.append(c_clean)

        ta_match = re.search(r'TAMIL_SUMMARY:\s*(.+)', response_text, re.IGNORECASE)
        if ta_match:
            tamil_summary = ta_match.group(1).strip()
            if tamil_summary.lower() in ["n/a", "none"]:
                tamil_summary = ""

        refutation_terms = ["refutes", "refuted", "debunked", "fabrication", "fabricated", "false", "baseless", "hoax", "unfounded", "rumor", "not confer", "no credible source"]
        full_reasoning_str = (response_text + " " + " ".join(reasoning_points)).lower()
        has_refutation_cues = any(term in full_reasoning_str for term in refutation_terms)
        has_evidence_false = any(e.get("verdict") in ["FALSE", "SCAM"] for e in evidence_items)
        has_evidence_misleading = any(e.get("verdict") == "MISLEADING" for e in evidence_items)
        has_evidence_true = any(e.get("verdict") in ["TRUE", "VERIFIED"] for e in evidence_items)

        if (has_evidence_false or has_refutation_cues) and category == "Supported":
            category = "Contradicted"
            deception_score = 0.92
        elif has_evidence_misleading and category in ["Supported", "Unverified"]:
            category = "Misleading"
            deception_score = 0.74
        elif has_evidence_true and category in ["Contradicted", "Misleading"]:
            category = "Supported"
            deception_score = 0.12

        return {
            "verdict_category": category,
            "deception_score": round(deception_score, 2),
            "reasoning_points": reasoning_points[:4],
            "tamil_summary": tamil_summary,
            "raw_llm_output": response_text
        }

    def _analytical_reasoning_fallback(self, claims: List[str], evidence_items: List[Dict[str, Any]], is_tamil: bool) -> Dict[str, Any]:
        has_false_verdict = any(e.get("verdict") in ["FALSE", "SCAM"] for e in evidence_items)
        has_misleading_verdict = any(e.get("verdict") == "MISLEADING" for e in evidence_items)
        has_true_verdict = any(e.get("verdict") in ["TRUE", "VERIFIED"] for e in evidence_items)

        if has_false_verdict:
            top_ev = next(e for e in evidence_items if e.get("verdict") in ["FALSE", "SCAM"])
            category = "Contradicted"
            deception_score = 0.92
            reasoning = [
                f"Directly contradicted by official authoritative record: '{top_ev.get('title')}' from {top_ev.get('source')}.",
                f"The verified evidence confirms: {top_ev.get('content')[:180]}...",
                "This claim matches a known viral fabrication or debunked hoax."
            ]
            ta_sum = f"இந்த தகவல் பொய்யானது என {top_ev.get('source')} அமைப்பு உறுதிப்படுத்தியுள்ளது."
        elif has_misleading_verdict:
            top_ev = next(e for e in evidence_items if e.get("verdict") == "MISLEADING")
            category = "Misleading"
            deception_score = 0.74
            reasoning = [
                f"Claim contains out-of-context or exaggerated elements according to {top_ev.get('source')}.",
                f"Contextual clarification: {top_ev.get('content')[:180]}...",
                "Key caveats, eligibility conditions, or scientific boundaries were omitted."
            ]
            ta_sum = f"இந்த தகவல் தவறாக வழிநடத்தும் வகையில் திரித்துக் கூறப்பட்டுள்ளது ({top_ev.get('source')})."
        elif has_true_verdict:
            top_ev = next(e for e in evidence_items if e.get("verdict") in ["TRUE", "VERIFIED"])
            category = "Supported"
            deception_score = 0.12
            reasoning = [
                f"Corroborated by verified record: '{top_ev.get('title')}' from {top_ev.get('source')}.",
                f"Official confirmation: {top_ev.get('content')[:180]}...",
                "The core assertions align with official publications."
            ]
            ta_sum = f"இந்த தகவல் உண்மையானது என அதிகாரப்பூர்வமாக உறுதிப்படுத்தப்பட்டுள்ளது ({top_ev.get('source')})."
        else:
            category = "Unverified"
            deception_score = 0.50
            reasoning = [
                "No identical claim or direct official fact-check was found in the indexed knowledge base.",
                "Assessment relies on machine learning linguistic classification and sensationalism indicators.",
                "Cross-referencing with primary official news sources is recommended."
            ]
            ta_sum = "இந்த செய்திக்கு நேரடி அதிகாரப்பூர்வ சரிபார்ப்பு தரவு கிடைக்கவில்லை; கூடுதல் ஆதாரங்கள் தேவை."

        return {
            "verdict_category": category,
            "deception_score": deception_score,
            "reasoning_points": reasoning,
            "tamil_summary": ta_sum if is_tamil else "",
            "raw_llm_output": "Generated via Analytical Reasoning Engine (RAG Synthesis)",
            "mode": "Analytical Engine (RAG Synthesis)"
        }
