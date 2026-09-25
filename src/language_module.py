import re
try:
    from langdetect import detect
except ImportError:
    detect = None

TANGLISH_NORMALIZATION_MAP = {
    "vadhanti": "vadhanthi",
    "vadhanthee": "vadhanthi",
    "vadhandhi": "vadhanthi",
    "poi": "poi",
    "poiya": "poi",
    "poiyanadhu": "poi",
    "thappu": "thavaru",
    "thavaraana": "thavaru",
    "thavaru": "thavaru",
    "unmai": "unmai",
    "unmaiya": "unmai",
    "seithi": "seithi",
    "seithigal": "seithi",
    "athirchi": "athirchi",
    "bayangaram": "bayangaram",
    "share": "share",
    "pannunga": "pannunga",
    "panunga": "pannunga",
    "anupunga": "anupunga",
    "kelunga": "kelunga",
    "parunga": "parunga",
    "nambadheenga": "nambadheenga",
    "nambadhinga": "nambadheenga",
    "iniku": "iniki",
    "inniku": "iniki",
    "netru": "netru",
    "naalaiku": "naalaiku",
    "udane": "udane",
    "seekiram": "udane",
    "panam": "panam",
    "kaasu": "panam",
    "kadan": "kadan",
    "roobai": "rubai",
    "kedaikum": "kedaikum",
    "kudupaanga": "kudupaanga",
    "vangunga": "vangunga"
}

TANGLISH_VOCABULARY = {
    "iniku", "inniku", "netru", "naalaiku", "udane", "vadhanthi", "vadhanti",
    "unmai", "seithi", "athirchi", "pannunga", "panunga", "anupunga",
    "kedaikum", "kudupaanga", "vandhuchu", "poiya", "thavaru", "iruku",
    "irukku", "evlo", "epdi", "yenna", "enna", "theriyuma", "paathingala",
    "nambatheenga", "nambadhinga", "arasanadhu", "makkale", "thozhar"
}

def detect_language(text: str) -> dict:
    if not text or not text.strip():
        return {"language": "en", "label": "English", "is_tamil": False, "is_tanglish": False}

    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    total_letters = len(re.findall(r'[a-zA-Z\u0B80-\u0BFF]', text)) or 1
    tamil_ratio = tamil_chars / total_letters

    if tamil_ratio > 0.20:
        return {
            "language": "ta",
            "label": "Tamil (தமிழ்)",
            "is_tamil": True,
            "is_tanglish": False,
            "tamil_ratio": round(tamil_ratio, 2)
        }

    words = re.findall(r'[a-zA-Z]+', text.lower())
    tanglish_matches = [w for w in words if w in TANGLISH_VOCABULARY or w in TANGLISH_NORMALIZATION_MAP]
    
    if len(tanglish_matches) >= 2 or (len(words) > 0 and len(tanglish_matches) / len(words) > 0.15):
        return {
            "language": "tanglish",
            "label": "Tanglish (Tamil in English Script)",
            "is_tamil": False,
            "is_tanglish": True,
            "matched_tanglish_tokens": tanglish_matches
        }

    detected = "en"
    if detect:
        try:
            detected = detect(text)
        except Exception:
            detected = "en"

    label = "English" if detected == "en" else f"Other ({detected})"
    return {
        "language": detected,
        "label": label,
        "is_tamil": False,
        "is_tanglish": False
    }

def normalize_multilingual_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip()
    words = cleaned.split()
    normalized_words = []
    for word in words:
        clean_w = re.sub(r'[^\w\s\u0B80-\u0BFF]', '', word.lower())
        replacement = TANGLISH_NORMALIZATION_MAP.get(clean_w, word)
        normalized_words.append(replacement)
    return " ".join(normalized_words)
