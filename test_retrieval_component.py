import json
from langchain_retrieval.rag_retriever import retrieve_evidence

test_cases = [
    "What is GSLV F17?",
    "What has India's space agency recently achieved?",
    "Did ISRO's recent satellite mission achieve what this article claims?"
]

for q in test_cases:
    print("=" * 70)
    print(f"INPUT: {q}")
    res = retrieve_evidence(q, k=2)
    print(f"STRATEGY: {res['strategy']}")
    print(f"ROUTER REASON: {res['router_reason']}")
    print(f"CLAIMS: {res['claims']}")
    print(f"EVIDENCE COUNT: {len(res['evidence'])}")
    for i, ev in enumerate(res['evidence']):
        print(f"  [{i+1}] Source: {ev['source']} ({ev['source_type']}) | Score: {ev['score']}")
        print(f"      URL: {ev['url']}")
        print(f"      Content: {ev['content'][:140]}...")
