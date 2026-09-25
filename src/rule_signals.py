import re
from typing import Dict, Any, List

RULE_PATTERNS = {
    "urgent": {
        "weight": 0.25,
        "description": "Urgency and pressure tactics (viral forwarding pressure)",
        "patterns": [
            r'\b(?:breaking\s*news|urgent\s*alert|emergency\s*alert|share\s*immediately|forward\s*immediately)\b',
            r'\b(?:before\s*(?:it\s*gets\s*)?deleted|do\s*not\s*ignore|forward\s*to\s*(?:all|everyone))\b',
            r'\b(?:pass\s*this\s*on|share\s*this\s*to\s*\d+\s*(?:people|groups|friends))\b',
            r'\b(?:warn\s*your\s*family|alert\s*all\s*citizens)\b',
            r'அவசர\s*செய்தி|அவசரம்|உடனே\s*பகிருங்கள்|உடனே\s*பாருங்கள்|அனைவருக்கும்\s*பகிருங்கள்',
            r'உடனே\s*ஷேர்\s*பண்ணுங்க|தவறவிடாதீர்கள்|உடனே\s*அனுப்புங்கள்|இப்போதே\s*பகிருங்கள்',
            r'\b(?:udane\s*share|share\s*pannunga|forward\s*pannunga|udane\s*anupunga)\b'
        ]
    },
    "shocking": {
        "weight": 0.25,
        "description": "Sensationalism and hyperbole (clickbait triggers)",
        "patterns": [
            r'\b(?:shocking\s*truth|unbelievable|mind\s*blowing|miracle\s*cure|secret\s*revealed)\b',
            r'\b(?:doctors\s*(?:are\s*)?(?:hiding|stunned|shocked)|hidden\s*truth|they\s*don\'?t\s*want\s*you\s*to\s*know)\b',
            r'\b(?:100%\s*(?:guaranteed|cure|proven|true)|magical\s*cure|cures\s*all\s*diseases)\b',
            r'\b(?:exposed\s*finally|conspiracy|jaw\s*dropping|bizarre\s*secret)\b',
            r'அதிர்ச்சி\s*தகவல்|அதிர்ச்சி|ரகசியம்\s*அம்பலம்|மருத்துவர்கள்\s*மறைத்த|அதிசய\s*மருந்து',
            r'100%\s*உண்மை|மிரள\s*வைக்கும்|பகிரங்க\s*உண்மை|அதிசயம்|மர்மம்\s*விலகியது',
            r'\b(?:athirchi\s*seithi|athirchi\s*thakaval|100%\s*unmai|ragasiyam\s*ambalam)\b'
        ]
    },
    "sponsored_scam": {
        "weight": 0.25,
        "description": "Commercial scam, phishing, or click-fraud indicators",
        "patterns": [
            r'\b(?:sponsored|click\s*here|claim\s*now|free\s*gift|win\s*cash|lottery\s*winner)\b',
            r'\b(?:earn\s*(?:up\s*to)?\s*\d+\s*(?:daily|per\s*day|from\s*home)|work\s*from\s*home\s*earn)\b',
            r'\b(?:send\s*otp|verify\s*bank\s*details|electricity\s*disconnected\s*tonight)\b',
            r'\b(?:free\s*recharge|free\s*laptop\s*scheme|click\s*the\s*link\s*below)\b',
            r'இங்கே\s*கிளிக்\s*செய்யவும்|இலவசமாக\s*பெற|பரிசு\s*வெல்லுங்கள்|இலவச\s*ரீசார்ஜ்',
            r'மின்\s*இணைப்பு\s*துண்டிக்கப்படும்|இலவச\s*லேப்டாப்|உடனே\s*பதிவு\s*செய்யுங்கள்',
            r'\b(?:click\s*panni|link\s*click|free\s*recharge|free\s*laptop)\b'
        ]
    },
    "whatsapp_chain": {
        "weight": 0.15,
        "description": "Viral social media forwarding signatures",
        "patterns": [
            r'\b(?:forwarded\s*as\s*received|received\s*on\s*whatsapp|forward\s*as\s*received)\b',
            r'\b(?:as\s*seen\s*on\s*whatsapp|forwarded\s*message)\b',
            r'\b(?:please\s*circulate|share\s*maximum)\b',
            r'வாட்ஸ்அப்\s*தகவல்|பகிர்ந்தவர்|வந்த\s*செய்தி'
        ]
    }
}

def analyze_rule_signals(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "rule_score": 0.0,
            "risk_level": "Low",
            "detected_signals": [],
            "caps_ratio": 0.0,
            "exclamation_count": 0,
            "indicators": []
        }

    detected_signals: List[Dict[str, Any]] = []
    total_rule_score = 0.0
    text_lower = text.lower()

    for category, config in RULE_PATTERNS.items():
        matches = []
        for pattern in config["patterns"]:
            found = re.findall(pattern, text_lower, re.IGNORECASE)
            if found:
                matches.extend(found)
                
        if matches:
            unique_matches = list(set([m if isinstance(m, str) else m[0] for m in matches]))
            category_score = min(config["weight"] * len(unique_matches), config["weight"] * 1.5)
            total_rule_score += category_score
            detected_signals.append({
                "category": category,
                "description": config["description"],
                "matches": unique_matches,
                "score_impact": round(category_score, 3)
            })

    words = re.findall(r'\b[A-Za-z]+\b', text)
    caps_words = [w for w in words if w.isupper() and len(w) > 2 and w not in {"WHO", "NASA", "RBI", "ISRO", "PIB", "TNEB", "UNESCO", "ICMR", "TANGEDCO", "AI", "PM", "USA", "UK"}]
    caps_ratio = len(caps_words) / len(words) if words else 0.0

    if caps_ratio > 0.15:
        caps_score = min(0.20, caps_ratio * 0.4)
        total_rule_score += caps_score
        detected_signals.append({
            "category": "ALL_CAPS",
            "description": f"High proportion of ALL-CAPS words ({round(caps_ratio*100, 1)}%) often associated with shouting or clickbait",
            "matches": caps_words[:6],
            "score_impact": round(caps_score, 3)
        })

    exclamations = len(re.findall(r'!{2,}|\?{2,}|!\?|\?!', text))
    if exclamations > 0:
        punct_score = min(0.15, exclamations * 0.05)
        total_rule_score += punct_score
        detected_signals.append({
            "category": "excessive_punctuation",
            "description": f"Detected {exclamations} multiple exclamation/question mark sequences (sensationalist emphasis)",
            "matches": [f"{exclamations} instances of '!!' or '??'"],
            "score_impact": round(punct_score, 3)
        })

    final_rule_score = min(1.0, round(total_rule_score, 2))

    if final_rule_score >= 0.65:
        risk_level = "High"
    elif final_rule_score >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    indicators = []
    for sig in detected_signals:
        indicators.append(f"[{sig['category'].upper()}] {sig['description']}: {', '.join(sig['matches'][:3])}")

    return {
        "rule_score": final_rule_score,
        "risk_level": risk_level,
        "detected_signals": detected_signals,
        "caps_ratio": round(caps_ratio, 2),
        "exclamation_count": exclamations,
        "indicators": indicators
    }
