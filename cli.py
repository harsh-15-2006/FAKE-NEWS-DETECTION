#!/usr/bin/env python
"""
SATYA — command-line entry point.

Every threshold, weight and toggle is a flag with a sensible default from
config/settings.yaml. Nothing behavioural is hardcoded in src/.

Examples
--------
  python cli.py --text "Indha marundhu saapttaa sugar level 3 naalla normal aagidum"
  python cli.py --text "..." --backend offline --no-llm
  python cli.py --file article.txt --mode article --format json --out out/
  python cli.py --text "..." --llm-model llama3.2:latest --router-strategy semantic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import apply_overrides, load_config, resolve_api_key
from src.models import Verdict
from src.pipeline import analyze

BADGE = {
    Verdict.SUPPORTED.value: "[SUPPORTED]",
    Verdict.REFUTED.value: "[REFUTED]",
    Verdict.INSUFFICIENT_EVIDENCE.value: "[INSUFFICIENT EVIDENCE]",
    Verdict.ESCALATE.value: "[ESCALATE - HUMAN REVIEW]",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="satya",
        description="Claim-level misinformation verification (Tamil/Hindi/English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("input")
    src.add_argument("--text", help="claim or message text to verify")
    src.add_argument("--file", help="path to a .txt file to verify")
    src.add_argument("--mode", choices=["claim", "article"], default=None,
                     help="claim = short forward; article = multi-claim document")

    lang = p.add_argument_group("language")
    lang.add_argument("--langs", help="comma list of retrieval languages, e.g. ta,hi,en")
    lang.add_argument("--transliterate-romanized", dest="translit",
                      choices=["true", "false"], default=None)
    lang.add_argument("--translit-backend", choices=["auto", "rule", "llm", "off"],
                      default=None)
    lang.add_argument("--pivot-lang", default=None)

    ret = p.add_argument_group("retrieval")
    ret.add_argument("--backend", choices=["api", "offline", "hybrid"], default=None)
    ret.add_argument("--dual-path", choices=["true", "false"], default=None)
    ret.add_argument("--api-key", default=None,
                     help="overrides GOOGLE_FACTCHECK_API_KEY and .apikey")
    ret.add_argument("--topk", type=int, default=None)
    ret.add_argument("--rrf-k", type=int, default=None)
    ret.add_argument("--prior-debunk-window", type=int, default=None,
                     help="days back to search the debunk corpus")
    ret.add_argument("--resurface-gap-days", type=int, default=None)
    ret.add_argument("--evidence-min-sources", type=int, default=None)
    ret.add_argument("--router-strategy",
                     choices=["auto", "keyword", "semantic", "hybrid"], default=None)

    claims = p.add_argument_group("claims")
    claims.add_argument("--checkworthy-threshold", type=float, default=None)
    claims.add_argument("--max-claims", type=int, default=None)

    llm = p.add_argument_group("llm")
    llm.add_argument("--llm-model", default=None)
    llm.add_argument("--llm-base-url", default=None)
    llm.add_argument("--llm-temperature", type=float, default=None)
    llm.add_argument("--llm-ctx", type=int, default=None)
    llm.add_argument("--llm-timeout", type=float, default=None)
    llm.add_argument("--llm-max-claims", type=int, default=None)
    llm.add_argument("--llm-only-when-ambiguous", choices=["true", "false"],
                     default=None)
    llm.add_argument("--no-llm", action="store_true",
                     help="disable the LLM stage entirely")

    ground = p.add_argument_group("grounding")
    ground.add_argument("--grounding", choices=["true", "false"], default=None)
    ground.add_argument("--grounding-strictness", choices=["exact", "normalized"],
                        default=None)
    ground.add_argument("--min-span-chars", type=int, default=None)

    scor = p.add_argument_group("scoring")
    scor.add_argument("--w-llm", type=float, default=None)
    scor.add_argument("--w-retrieval", type=float, default=None)
    scor.add_argument("--w-rule", type=float, default=None)
    scor.add_argument("--w-ml", type=float, default=None)
    scor.add_argument("--stance-threshold", type=float, default=None)
    scor.add_argument("--conflict-threshold", type=float, default=None)
    scor.add_argument("--never-auto-verdict", default=None,
                      help="comma list of categories that always escalate")

    out = p.add_argument_group("output")
    out.add_argument("--format", choices=["text", "json"], default=None)
    out.add_argument("--out", default=None, help="directory for the JSON report")
    out.add_argument("--quiet", action="store_true", help="suppress progress lines")

    cfgs = p.add_argument_group("config files")
    cfgs.add_argument("--settings", default=None)
    cfgs.add_argument("--categories", default=None)
    cfgs.add_argument("--lexicon", default=None)
    return p


def _b(v: str | None) -> bool | None:
    return None if v is None else v == "true"


def _csv(v: str | None) -> list[str] | None:
    return None if v is None else [x.strip() for x in v.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.text:
        text = args.text
    elif args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            return 2
        text = path.read_text(encoding="utf-8")
    else:
        print("ERROR: provide --text or --file", file=sys.stderr)
        return 2

    cfg = load_config(args.settings, args.categories, args.lexicon)

    overrides = {
        "languages.supported": _csv(args.langs),
        "languages.transliterate_romanized": _b(args.translit),
        "languages.translit_backend": args.translit_backend,
        "languages.pivot": args.pivot_lang,
        "retrieval.backend": args.backend,
        "retrieval.dual_path": _b(args.dual_path),
        "retrieval.topk": args.topk,
        "retrieval.rrf_k": args.rrf_k,
        "retrieval.prior_debunk_window_days": args.prior_debunk_window,
        "retrieval.resurface_gap_days": args.resurface_gap_days,
        "retrieval.evidence_min_sources": args.evidence_min_sources,
        "claims.checkworthy_threshold": args.checkworthy_threshold,
        "claims.max_claims_per_doc": args.max_claims,
        "llm.model": args.llm_model,
        "llm.base_url": args.llm_base_url,
        "llm.temperature": args.llm_temperature,
        "llm.num_ctx": args.llm_ctx,
        "llm.timeout_s": args.llm_timeout,
        "llm.max_claims": args.llm_max_claims,
        "llm.only_when_ambiguous": _b(args.llm_only_when_ambiguous),
        "llm.enabled": False if args.no_llm else None,
        "grounding.enabled": _b(args.grounding),
        "grounding.strictness": args.grounding_strictness,
        "grounding.min_span_chars": args.min_span_chars,
        "scoring.weights.llm": args.w_llm,
        "scoring.weights.retrieval": args.w_retrieval,
        "scoring.weights.rule": args.w_rule,
        "scoring.weights.ml": args.w_ml,
        "scoring.stance_threshold": args.stance_threshold,
        "scoring.conflict_threshold": args.conflict_threshold,
        "scoring.never_auto_verdict": _csv(args.never_auto_verdict),
        "output.format": args.format,
        "output.dir": args.out,
    }
    cfg = apply_overrides(cfg, overrides)
    if args.router_strategy:
        cfg["_forced_strategy"] = args.router_strategy

    api_key = resolve_api_key(args.api_key)
    if not api_key and cfg["retrieval"]["backend"] in ("api", "hybrid"):
        print("NOTE: no API key found (.apikey file / GOOGLE_FACTCHECK_API_KEY / "
              "--api-key). Live fact-check search will be skipped; the offline "
              "corpus will still be used.\n", file=sys.stderr)

    progress = None if args.quiet else (lambda m: print(m, file=sys.stderr))
    report = analyze(text, cfg, api_key, progress)

    if cfg["output"]["format"] == "json":
        payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
        outdir = cfg["output"].get("dir")
        if outdir:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            dest = Path(outdir) / "report.json"
            dest.write_text(payload, encoding="utf-8")
            print(f"\nreport written: {dest}")
        else:
            print(payload)
        return 0

    # ---- human-readable ------------------------------------------------ #
    L = report.language
    print("\n" + "=" * 70)
    print("SATYA REPORT")
    print("=" * 70)
    print(f"language      : {L.lang_code}  (romanised={L.was_romanized})")
    print(f"transliteration: {L.translit_method}")
    print(f"translation    : {L.translate_method}")
    if L.was_romanized:
        print(f"native script  : {L.native_script}")
    print(f"english pivot  : {L.pivot_en}")
    for n in L.notes:
        print(f"  note: {n}")

    for r in report.claims:
        print("\n" + "-" * 70)
        print(f"CLAIM {r.claim.idx}: {r.claim.text_native}")
        if r.claim.text_en != r.claim.text_native:
            print(f"  (en) {r.claim.text_en}")
        print(f"  check-worthiness : {r.claim.checkworthy}")
        print(f"  category         : {r.signals.category} "
              f"({r.signals.category_confidence:.2f})")
        print(f"  strategy         : {r.evidence.strategy_used}")
        print(f"  rule score       : {r.signals.rule_score}  {r.signals.rule_hits}")
        print(f"  ml prior         : {r.signals.ml_score}  ({r.signals.ml_basis})")
        print(f"\n  {BADGE.get(r.verdict.value, r.verdict.value)}  "
              f"confidence={r.confidence}  gate={r.gate_fired}")
        print(f"  {r.reasoning}")

        if r.grounding:
            status = "PASSED" if r.grounding.passed else "REJECTED"
            print(f"\n  GROUNDING CHECK: {status}")
            for reason in r.grounding.reasons:
                print(f"    - {reason}")

        if r.evidence.passages:
            print("\n  EVIDENCE:")
            for p in r.evidence.passages:
                tag = " [SYNTHETIC SAMPLE]" if p.synthetic else ""
                res = f" [RESURFACED +{p.resurface_gap_days}d]" if p.resurfaced else ""
                print(f"    [{p.pid}] {p.publisher} — {p.rating}{tag}{res}")
                if p.url:
                    print(f"        {p.url}")
        for e in r.evidence.errors:
            print(f"    ! {e}")

    a = report.aggregate
    print("\n" + "=" * 70)
    print(f"claims={a['n_claims']}  {a['verdict_counts']}")
    print(f"llm={a['llm_model']} calls={a['llm_calls']}  "
          f"grounding_rejections={a['grounding_rejections']}")
    print(f"api_key_present={a['api_key_present']}  "
          f"synthetic_evidence_used={a['synthetic_evidence_used']}")
    print(f"elapsed={a['elapsed_s']}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
