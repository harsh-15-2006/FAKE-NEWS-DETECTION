import re
from typing import Dict, Any

# Signals for exact-match detection
ORGANIZATIONS = ["ISRO", "WHO", "NASA", "IRCH", "MNRE", "TNEB", "PIB", "ICMR"]
MISSION_IDENTIFIERS = [
    r'\b(?:GSLV[\s\-_]*F\d+)\b',
    r'\b(?:PSLV[\s\-_]*C\d+)\b',
    r'\b(?:LVM3[\s\-_]*M\d+)\b',
    r'\b(?:BlueBird(?:\s*Block\s*\d+)?)\b',
    r'\b(?:EOS[\s\-_]*\d+)\b',
    r'\b(?:Chandrayaan(?:\s*-\s*\d+)?)\b',
    r'\b(?:Aditya[\s\-_]*L\d+)\b',
    r'\b(?:Gaganyaan)\b'
]
DATE_PATTERNS = [
    r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    r'\b(?:202[4-9]|203[0-9])\b',
    r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}\b'
]
CONTEXTUAL_QUESTION_PATTERNS = [
    r'\b(?:did|does|is|are|was|were|has|have|can|will|should)\b.+\b(?:achieve|claim|true|false|real|fake|happen|succeed|confirm|cause)\b',
    r'\b(?:what|why|how|explain|describe)\b.+\b(?:recent|achievement|progress|advance|difference|purpose)\b'
]

def route_query(query: str) -> Dict[str, str]:
    """
    Lightweight Search Router.
    Routes queries to:
    - 'keyword' (exact-match signals: mission IDs, acronyms, dates, quotes)
    - 'semantic' (broad, conceptual queries)
    - 'hybrid' (specific entities combined with contextual / evaluative meaning)
    """
    if not query or not query.strip():
        return {
            "strategy": "semantic",
            "reason": "Empty or default query; defaulting to semantic vector search."
        }

    q_lower = query.lower()
    
    # 1. Detect exact mission identifiers
    matched_missions = []
    for pattern in MISSION_IDENTIFIERS:
        m = re.findall(pattern, query, re.IGNORECASE)
        if m:
            matched_missions.extend(m)

    # 2. Detect organizations and acronyms
    matched_orgs = [org for org in ORGANIZATIONS if re.search(r'\b' + org + r'\b', query, re.IGNORECASE)]

    # 3. Detect dates
    matched_dates = []
    for dp in DATE_PATTERNS:
        d = re.findall(dp, query, re.IGNORECASE)
        if d:
            matched_dates.extend(d)

    # 4. Detect quoted phrases
    has_quotes = bool(re.search(r'["\'][^"\']+["\']', query))

    # 5. Detect contextual / inferential queries
    is_contextual = any(re.search(pat, q_lower) for pat in CONTEXTUAL_QUESTION_PATTERNS)
    has_comparative_words = any(w in q_lower for w in ["claim", "claims", "achieve", "recently", "unusual", "strange", "status", "overview", "cooperation"])

    has_exact_signals = bool(matched_missions or matched_orgs or matched_dates or has_quotes)

    # Decision Matrix
    if has_exact_signals and (is_contextual or has_comparative_words or len(query.split()) > 7):
        entity_desc = []
        if matched_missions:
            entity_desc.append(f"mission '{matched_missions[0]}'")
        if matched_orgs:
            entity_desc.append(f"organization '{matched_orgs[0]}'")
        if matched_dates:
            entity_desc.append(f"date '{matched_dates[0]}'")
            
        return {
            "strategy": "hybrid",
            "reason": f"The query contains specific entities ({', '.join(entity_desc)}) and requires contextual matching."
        }
    elif has_exact_signals:
        signals = []
        if matched_missions:
            signals.append(f"mission identifier '{matched_missions[0]}'")
        if matched_orgs:
            signals.append(f"organization '{matched_orgs[0]}'")
        if matched_dates:
            signals.append("date pattern")
        if has_quotes:
            signals.append("exact quoted phrase")
            
        return {
            "strategy": "keyword",
            "reason": f"Specific {', '.join(signals)} detected for exact lexical match."
        }
    else:
        return {
            "strategy": "semantic",
            "reason": "Broad or conceptual query without exact identifiers; semantic vector search provides optimal recall."
        }

if __name__ == "__main__":
    test_queries = [
        "What is GSLV F17?",
        "What has India's space agency recently achieved?",
        "Did ISRO's recent satellite mission achieve what this article claims?"
    ]
    for q in test_queries:
        print(f"Query: '{q}' -> {route_query(q)}")
