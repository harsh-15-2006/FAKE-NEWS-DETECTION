import re
from typing import List, Dict, Any
from src.language_module import detect_language, normalize_multilingual_text

CLAIM_TRIGGER_VERBS = [
    "claims", "discovered", "cured", "cures", "proven", "proves", "ordered",
    "banned", "declared", "announced", "warns", "confirms", "allocated",
    "cancels", "shutdown", "caused", "eradicates", "gives free", "free",
    "secretly", "advises", "leaked", "confiscated", "transferred",
    "அறிவித்துள்ளது", "உத்தரவிட்டுள்ளது", "தெரிவித்துள்ளது", "உறுதிசெய்துள்ளது",
    "ரத்து", "தடை", "வழங்கப்படும்", "கண்டுபிடிப்பு", "குணப்படுத்தும்",
    "நிரூபணம்", "துண்டிக்கப்படும்", "அதிர்ச்சி", "பொய்", "உண்மை"
]

def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', raw_text)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_claims(text: str) -> List[str]:
    if not text:
        return []
    raw_sentences = re.split(r'(?<=[.!?\n])\s+', text)
    claims = []

    for s in raw_sentences:
        clean_s = s.strip()
        if len(clean_s) < 15:
            continue
        lower_s = clean_s.lower()
        has_trigger = any(verb in lower_s for verb in CLAIM_TRIGGER_VERBS)
        has_digits = bool(re.search(r'\d+', clean_s))
        has_quotes = '"' in clean_s or "'" in clean_s
        
        if has_trigger or has_digits or has_quotes or len(clean_s) > 40:
            claims.append(clean_s)

    if not claims and text:
        claims = [text[:250]]

    return claims

def extract_entities_and_keywords(text: str) -> Dict[str, Any]:
    entities = {
        "organizations": [],
        "monetary_amounts": [],
        "dates_numbers": [],
        "keywords": []
    }
    
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
    entities["organizations"] = list(set(acronyms))
    
    amounts = re.findall(r'(?:Rs\.?|INR|ரூ\.?|\$)\s*[\d,]+(?:\.\d+)?|\b[\d,]+\s*(?:rupees|crore|lakh|ரூபாய்)\b', text, re.IGNORECASE)
    entities["monetary_amounts"] = list(set(amounts))
    
    numbers = re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\d+%\b|\d+(?:\.\d+)?)\b', text)
    entities["dates_numbers"] = list(set(numbers[:8]))
    
    stop_words = {
        "this", "that", "with", "from", "have", "were", "been", "will", "what",
        "they", "their", "about", "which", "there", "would", "could", "should",
        "into", "more", "than", "them", "when", "some", "news", "article",
        "இந்த", "அந்த", "என்று", "ஆனால்", "மற்றும்", "ஒரு", "செய்தி"
    }
    
    words = re.findall(r'[\w\u0B80-\u0BFF]{4,}', text.lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w not in stop_words and not w.isdigit():
            freq[w] = freq.get(w, 0) + 1
            
    sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    entities["keywords"] = [w for w, _ in sorted_keywords[:10]]
    
    return entities

def preprocess_news(raw_text: str) -> Dict[str, Any]:
    cleaned = clean_text(raw_text)
    lang_info = detect_language(cleaned)
    normalized = normalize_multilingual_text(cleaned)
    claims = extract_claims(cleaned)
    entities_keywords = extract_entities_and_keywords(cleaned)
    
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "normalized_text": normalized,
        "language_info": lang_info,
        "extracted_claims": claims,
        "entities": entities_keywords["organizations"],
        "monetary_amounts": entities_keywords["monetary_amounts"],
        "dates_numbers": entities_keywords["dates_numbers"],
        "top_keywords": entities_keywords["keywords"]
    }
