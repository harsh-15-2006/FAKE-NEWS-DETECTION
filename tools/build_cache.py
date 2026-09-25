#!/usr/bin/env python
"""
Build the OFFLINE EVIDENCE CACHE from the live Google Fact Check Tools API.

This fetches REAL ClaimReview records and writes them to
cache/claimreview.json, so the pipeline keeps working with the wifi switched
off. Run it once while you have network; it is the single step that makes the
demo survive a venue outage.

    python tools/build_cache.py
    python tools/build_cache.py --langs en,ta,hi --per-query 10
    python tools/build_cache.py --queries-file tools/seed_queries.txt

Records written here carry no _synthetic flag, so they are treated as real
evidence and the dashboard's synthetic-data warning stays off.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_api_key  # noqa: E402

ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Broad seed topics. The API matches across claim text and review text, so
# broad terms return more than a specific debunked sentence would.
DEFAULT_QUERIES = [
    "India", "vaccine", "covid", "health cure", "medicine", "cancer",
    "election India", "government scheme", "prime minister", "police",
    "viral video India", "WhatsApp forward", "old video", "fake photo",
    "Tamil Nadu", "cricket", "actor", "bank fraud", "UPI", "lottery",
    "temple", "riot", "protest", "currency note", "Aadhaar",
]


def fetch(query: str, lang: str, key: str, page_size: int,
          max_age_days: int, timeout: float) -> tuple[list[dict], str | None]:
    params = {
        "query": query,
        "languageCode": lang,
        "pageSize": page_size,
        "maxAgeDays": max_age_days,
        "key": key,
    }
    try:
        r = requests.get(ENDPOINT, params=params, timeout=timeout)
        if r.status_code == 403:
            return [], "403 - key invalid or Fact Check Tools API not enabled"
        if r.status_code == 429:
            return [], "429 - quota exceeded"
        r.raise_for_status()
        return r.json().get("claims", []), None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)


def flatten(raw_claims: list[dict]) -> list[dict]:
    out = []
    for c in raw_claims:
        text = c.get("text", "") or ""
        claim_date = c.get("claimDate")
        claimant = c.get("claimant", "")
        for rev in c.get("claimReview", []) or []:
            out.append({
                "text": text,
                "claimant": claimant,
                "publisher": (rev.get("publisher") or {}).get("name", "unknown"),
                "publisherSite": (rev.get("publisher") or {}).get("site", ""),
                "url": rev.get("url", ""),
                "title": rev.get("title", ""),
                "textualRating": rev.get("textualRating", ""),
                "claimDate": claim_date,
                "reviewDate": rev.get("reviewDate"),
                "languageCode": rev.get("languageCode", ""),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the offline ClaimReview cache.")
    ap.add_argument("--langs", default="en,ta,hi")
    ap.add_argument("--per-query", type=int, default=10)
    ap.add_argument("--max-age-days", type=int, default=3650)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--sleep", type=float, default=0.2, help="pause between calls")
    ap.add_argument("--queries-file", default=None,
                    help="newline-separated queries; overrides the built-in list")
    ap.add_argument("--out", default=None, help="output path")
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    cfg = load_config()
    key = resolve_api_key(args.api_key)
    if not key:
        print("ERROR: no API key. Save it to D:\\FAKE-NEWS-DETECTION\\.apikey "
              "or set GOOGLE_FACTCHECK_API_KEY or pass --api-key.",
              file=sys.stderr)
        return 2

    if args.queries_file:
        queries = [q.strip() for q in
                   Path(args.queries_file).read_text(encoding="utf-8").splitlines()
                   if q.strip()]
    else:
        queries = DEFAULT_QUERIES

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    out_path = Path(args.out) if args.out else (
        ROOT / cfg["retrieval"].get("offline_cache", "cache/claimreview.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    records: list[dict] = []
    errors: list[str] = []
    total_calls = len(queries) * len(langs)
    done = 0

    for q in queries:
        for lang in langs:
            done += 1
            raw, err = fetch(q, lang, key, args.per_query,
                             args.max_age_days, args.timeout)
            if err:
                errors.append(f"{q}[{lang}]: {err}")
                print(f"  [{done}/{total_calls}] {q} ({lang}) -> ERROR {err}")
                if "403" in err:
                    print("\nSTOPPING: the key or the API enablement is the "
                          "problem, retrying will not help.", file=sys.stderr)
                    return 3
            else:
                added = 0
                for rec in flatten(raw):
                    dedupe_key = rec["url"] or rec["text"][:150]
                    if dedupe_key in seen:
                        continue
                    seen.add(dedupe_key)
                    records.append(rec)
                    added += 1
                print(f"  [{done}/{total_calls}] {q} ({lang}) -> "
                      f"{len(raw)} claims, +{added} new")
            time.sleep(args.sleep)

    payload = {
        "_README": "REAL ClaimReview records fetched from the Google Fact Check "
                   "Tools API. Not synthetic.",
        "_built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_queries": queries,
        "_langs": langs,
        "_errors": errors,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    by_lang: dict[str, int] = {}
    for r in records:
        by_lang[r.get("languageCode", "?")] = by_lang.get(r.get("languageCode", "?"), 0) + 1

    print(f"\nwrote {len(records)} unique records -> {out_path}")
    print(f"by language: {by_lang}")
    if errors:
        print(f"{len(errors)} query errors (see _errors in the file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
