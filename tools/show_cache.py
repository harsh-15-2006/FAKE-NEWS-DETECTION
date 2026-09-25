#!/usr/bin/env python
"""
Inspect the offline evidence cache and pick good demo claims.

    python tools/show_cache.py
    python tools/show_cache.py --lang ta
    python tools/show_cache.py --rated-only --limit 20
    python tools/show_cache.py --resurfaced

Use this to choose demo inputs that are GUARANTEED to have real evidence
behind them — paste a printed claim straight into cli.py or the dashboard.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REFUTING = {"false", "fake", "incorrect", "misleading", "mostly false",
            "pants on fire", "altered", "miscaptioned", "no evidence",
            "unproven", "distorted", "manipulated", "fabricated", "hoax"}


def parse_date(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "cache" / "claimreview.json"))
    ap.add_argument("--lang", default=None, help="filter: en | ta | hi")
    ap.add_argument("--rated-only", action="store_true",
                    help="only records with a clearly refuting rating")
    ap.add_argument("--resurfaced", action="store_true",
                    help="only records where claimDate is long before reviewDate")
    ap.add_argument("--gap-days", type=int, default=180)
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    path = Path(args.cache)
    if not path.exists():
        print(f"cache not found: {path}\nRun: python tools/build_cache.py",
              file=sys.stderr)
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", data if isinstance(data, list) else [])

    by_lang: dict[str, int] = {}
    by_pub: dict[str, int] = {}
    for r in records:
        by_lang[r.get("languageCode", "?")] = by_lang.get(r.get("languageCode", "?"), 0) + 1
        by_pub[r.get("publisher", "?")] = by_pub.get(r.get("publisher", "?"), 0) + 1

    print(f"\ncache: {path}")
    print(f"total records: {len(records)}")
    print(f"by language  : {by_lang}")
    print("top publishers:")
    for pub, n in sorted(by_pub.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {n:>4}  {pub}")

    sel = records
    if args.lang:
        sel = [r for r in sel if r.get("languageCode") == args.lang]
    if args.rated_only:
        sel = [r for r in sel
               if any(t in (r.get("textualRating") or "").casefold() for t in REFUTING)]
    if args.resurfaced:
        out = []
        for r in sel:
            dc, dr = parse_date(r.get("claimDate")), parse_date(r.get("reviewDate"))
            if dc and dr and (dr - dc).days >= args.gap_days:
                r = {**r, "_gap": (dr - dc).days}
                out.append(r)
        sel = sorted(out, key=lambda r: -r["_gap"])

    print(f"\nmatching records: {len(sel)}  (showing {min(args.limit, len(sel))})")
    print("=" * 72)
    for r in sel[: args.limit]:
        gap = f"  [RESURFACED +{r['_gap']}d]" if "_gap" in r else ""
        print(f"\nlang={r.get('languageCode','?')}  "
              f"publisher={r.get('publisher','?')}  "
              f"rating={r.get('textualRating','?')}{gap}")
        print(f"CLAIM: {r.get('text','')}")
        if r.get("url"):
            print(f"  {r['url']}")
    print("\n" + "=" * 72)
    print("Copy a CLAIM line above and run:")
    print('  python cli.py --text "<paste claim here>" --format text')
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
