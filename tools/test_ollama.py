#!/usr/bin/env python
"""
Ollama diagnostic — finds exactly which request shape fails.

    python tools/test_ollama.py
    python tools/test_ollama.py --model llama3.2:latest --ctx 2048

Runs progressively harder requests and prints the real error body for each,
so we can see whether the problem is the model, JSON mode, the context size,
or memory.
"""

from __future__ import annotations

import argparse
import json
import time

import requests

BASE = "http://localhost:11434"


def show(label: str, ok: bool, detail: str, secs: float) -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"\n[{mark}] {label}   ({secs:.1f}s)")
    print(f"       {detail}")


def post(path: str, payload: dict, timeout: float):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}{path}", json=payload, timeout=timeout)
        secs = time.time() - t0
        if r.status_code != 200:
            body = r.text[:600]
            return False, f"HTTP {r.status_code}: {body}", secs
        data = r.json()
        content = (
            data.get("message", {}).get("content")
            or data.get("response", "")
        )
        return True, f"replied {len(content)} chars: {content[:160]!r}", secs
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}", time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--num-gpu", type=int, default=0,
                    help="0 = CPU only (default); works around "
                         "'CUDA error: device kernel image is invalid'")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    print("=" * 66)
    print("OLLAMA DIAGNOSTIC")
    print("=" * 66)

    # 0 - server + models
    t0 = time.time()
    try:
        r = requests.get(f"{BASE}/api/tags", timeout=10)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", [])]
        show("server reachable", True, f"models: {models}", time.time() - t0)
    except Exception as exc:  # noqa: BLE001
        show("server reachable", False, str(exc), time.time() - t0)
        print("\nFIX: open a terminal and run `ollama serve`, leave it open.")
        return 2

    model = args.model or (models[0] if models else None)
    if not model:
        print("\nNo models available.")
        return 2
    print(f"\nusing model: {model}")

    results = {}
    details: list[str] = []
    gpu_opt = {"num_gpu": args.num_gpu}
    print(f"num_gpu     : {args.num_gpu}"
          f"{'  (CPU only)' if args.num_gpu == 0 else ''}")

    # 1 - simplest possible chat
    ok, detail, secs = post("/api/chat", {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word: hello"}],
        "stream": False,
        "options": dict(gpu_opt),
    }, args.timeout)
    show("1. plain chat", ok, detail, secs)
    results["plain"] = ok
    details.append(detail)

    # 2 - chat with JSON mode
    ok, detail, secs = post("/api/chat", {
        "model": model,
        "messages": [
            {"role": "system", "content": 'Reply with JSON only: {"ok": true}'},
            {"role": "user", "content": "go"},
        ],
        "stream": False,
        "format": "json",
        "options": dict(gpu_opt),
    }, args.timeout)
    show("2. chat + format=json", ok, detail, secs)
    results["json"] = ok
    details.append(detail)

    # 3 - chat with JSON mode and an explicit context size
    ok, detail, secs = post("/api/chat", {
        "model": model,
        "messages": [
            {"role": "system", "content": 'Reply with JSON only: {"ok": true}'},
            {"role": "user", "content": "go"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": args.ctx, **gpu_opt},
    }, args.timeout)
    show(f"3. chat + json + num_ctx={args.ctx}", ok, detail, secs)
    results["ctx"] = ok
    details.append(detail)

    # 4 - a realistic SATYA-sized stance prompt
    passages = "\n\n".join(
        f"[PASSAGE {i}] (publisher: Example{i}; rating: False)\n"
        f"Claim reviewed: A sample claim number {i} about a public event. "
        f"Rating by Example{i}: False. Review title: The claim is not supported "
        f"by available evidence."
        for i in range(3)
    )
    ok, detail, secs = post("/api/chat", {
        "model": model,
        "messages": [
            {"role": "system", "content":
                'Reply with JSON only: {"stance":"SUPPORT|CONTRADICT|'
                'NOT_IN_CONTEXT","cited_passage_ids":[0],"quoted_span":"exact '
                'text","reasoning":"one sentence"}'},
            {"role": "user", "content":
                f"CLAIM:\nA sample claim number 1 about a public event.\n\n"
                f"PASSAGES:\n{passages}"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.1, "num_ctx": args.ctx, **gpu_opt},
    }, args.timeout)
    show("4. realistic stance prompt (3 passages)", ok, detail, secs)
    results["stance"] = ok
    details.append(detail)

    blob = " ".join(details)
    cuda_problem = "CUDA" in blob or "device kernel image" in blob
    oom_problem = ("memory" in blob.lower() or "0xc0000005" in blob
                   or "not enough" in blob.lower())

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    if all(results.values()):
        print(f"All shapes work. Use:  --llm-model {model} "
              f"--llm-ctx {args.ctx} --llm-num-gpu {args.num_gpu}")
    elif cuda_problem and args.num_gpu != 0:
        print("CUDA error: Ollama is trying to use the GPU and its CUDA build")
        print("does not match this machine's driver or GPU architecture.")
        print("This is NOT a memory problem. Force CPU-only and retry:")
        print("    python tools/test_ollama.py --num-gpu 0")
    elif cuda_problem:
        print("CUDA error persists even with num_gpu=0, so the per-request")
        print("option is not stopping the server from initialising CUDA.")
        print("Restart the Ollama server with the GPU hidden:")
        print("    1) close the Ollama app / stop `ollama serve`")
        print('    2) $env:CUDA_VISIBLE_DEVICES=""')
        print("    3) $env:OLLAMA_NUM_GPU=0")
        print("    4) ollama serve")
        print("Then rerun this script in a NEW terminal.")
        print("If that still fails, run the demo with --no-llm; the pipeline")
        print("is fully functional without it and falls back to published")
        print("fact-checker ratings.")
    elif oom_problem:
        print("Looks like memory. Close other apps, or use a smaller context:")
        print("    python tools/test_ollama.py --ctx 1024 --num-gpu 0")
    elif not results["plain"]:
        print("Even a plain chat fails and the error is neither CUDA nor")
        print("obviously memory. Read the error body printed above — it is the")
        print("real message from llama-server. Run the demo with --no-llm in")
        print("the meantime; the pipeline is fully functional without it.")
    elif not results["json"]:
        print("Plain chat works but format=json fails. Run SATYA with")
        print("  --llm-json-mode off   (not yet a flag — tell Claude)")
    elif not results["ctx"]:
        print(f"json works but num_ctx={args.ctx} fails. Retry with a smaller")
        print("value:  python tools/test_ollama.py --ctx 1024")
    else:
        print("Short prompts work, the realistic one fails — the prompt is too")
        print("large for available memory. Use:  --topk 2 --llm-ctx 1024")
    print("=" * 66 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
