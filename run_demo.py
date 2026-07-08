"""docfinder CLI: run the funnel, or benchmark it against brute-force.

Easiest way to use this: edit the CONFIG block below and run

    python run_demo.py

Command-line flags still work and override the CONFIG values if you prefer:

    python run_demo.py --files 8000 --governing 20 --budget 250
    python run_demo.py --backend ollama --files 500      # real local models
    python run_demo.py --benchmark                        # brute-force vs funnel (timed)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

from docfinder import run
from docfinder.corpus import generate_corpus

# ==========================================================================
# CONFIG - edit these instead of typing command-line flags.
# ==========================================================================
RUN_BENCHMARK = False      # True = brute-force LLM vs funnel timing; False = funnel demo
BACKEND = "ollama"           # "mock" (offline, instant) or "ollama" (real local model)
MODEL = "qwen2.5:3b"       # Ollama model tag (used by ollama backend and benchmark)
SEED = 7                   # same seed = same synthetic Drive
OUT_DIR = "results"        # folder for saved reports ("" to skip saving)

# Funnel demo size
DEMO_FILES = 3000
DEMO_GOVERNING = 12        # Files that are important to find
DEMO_BUDGET = 200          # max LLM calls

# Benchmark size (kept small: the brute-force path hits the LLM on every file)
BENCH_FILES = 200
BENCH_GOVERNING = 6
BENCH_BUDGET = 25
# ==========================================================================


def _make_progress():
    """Return a callback(label, current, total) that draws a one-line progress
    bar on stderr. Redraws in place; starts a new line when the label changes."""
    state = {"label": None, "start": 0.0}

    def cb(label, current, total):
        if label != state["label"]:
            if state["label"] is not None:
                sys.stderr.write("\n")
            state["label"] = label
            state["start"] = time.time()
        width = 28
        frac = current / total if total else 1.0
        bar = "#" * int(width * frac) + "-" * (width - int(width * frac))
        elapsed = time.time() - state["start"]
        sys.stderr.write(f"\r  {label:<11} [{bar}] {current}/{total}  {elapsed:4.0f}s")
        sys.stderr.flush()
        if current >= total:
            sys.stderr.write("\n")
            sys.stderr.flush()

    return cb


def _save(out_dir, base, report, metrics):
    if not out_dir:
        return None
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, base + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return os.path.join(out_dir, base + ".txt")


# --------------------------------------------------------------------------
# Funnel demo
# --------------------------------------------------------------------------

def run_demo(args) -> None:
    files = args.files if args.files is not None else DEMO_FILES
    governing = args.governing if args.governing is not None else DEMO_GOVERNING
    budget = args.budget if args.budget is not None else DEMO_BUDGET

    verifier = embedder = None
    if args.backend == "ollama":
        from docfinder.ollama_llm import OllamaVerifier
        from docfinder.ollama_embeddings import OllamaEmbedder
        verifier = OllamaVerifier(model=args.model)
        embedder = OllamaEmbedder()
        print(f"[backend] real local models via Ollama: {args.model} + "
              f"nomic-embed-text\n          (~{budget} LLM calls + one embedding "
              f"per survivor; be patient on CPU)\n")

    corpus = generate_corpus(files, governing, seed=args.seed)
    # Progress bar only for the real backend; the mock is effectively instant.
    progress = _make_progress() if args.backend == "ollama" else None
    result = run(corpus, max_llm_calls=budget, seed=args.seed,
                 verifier=verifier, embedder=embedder, progress=progress)

    lines = [
        "=" * 56,
        f"  docfinder - {files} files, {governing} governing hidden",
        f"  backend={args.backend}  budget={budget}  seed={args.seed}",
        "=" * 56,
        result.summary(),
    ]
    if result.true_missed:
        lines.append("\nMissed governing docs (the hard tail):")
        for f in result.true_missed:
            lines.append(f"  - {f.name:<24} in {f.path:<20} "
                         f"score={result.scores[f.id]:.3f} kind={f.doc_kind}")
    else:
        lines.append("\nNo governing documents missed.")
    report = "\n".join(lines)
    print(report)

    metrics = {
        "mode": "demo",
        "params": {"files": files, "governing": governing, "budget": budget,
                   "seed": args.seed, "backend": args.backend, "model": args.model},
        "counts": {"n_total": result.n_total, "n_after_media": result.n_after_media,
                   "n_verified": result.n_verified, "n_audited": result.n_audited,
                   "surfaced": len(result.surfaced)},
        "recall": {"estimate": round(result.recall_estimate, 4),
                   "ci_low": round(result.recall_ci[0], 4),
                   "ci_high": round(result.recall_ci[1], 4),
                   "true_recall": result.true_recall,
                   "true_missed": len(result.true_missed)},
        "cost": {"funnel_tokens": result.funnel_tokens,
                 "naive_tokens": result.naive_tokens,
                 "cost_ratio": round(result.cost_ratio, 2)},
        "surfaced_docs": [f.name for f in result.surfaced],
        "missed_docs": [f.name for f in result.true_missed],
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"run_{files}f_{governing}g_{budget}b_{args.backend}_seed{args.seed}_{stamp}"
    path = _save(args.out, base, report, metrics)
    if path:
        print(f"\nsaved: {path}")


# --------------------------------------------------------------------------
# Benchmark: brute-force LLM over every file vs the funnel (real models, timed)
# --------------------------------------------------------------------------

def run_benchmark(args) -> None:
    from docfinder.ollama_embeddings import OllamaEmbedder
    from docfinder.ollama_llm import OllamaVerifier

    files = args.files if args.files is not None else BENCH_FILES
    governing = args.governing if args.governing is not None else BENCH_GOVERNING
    budget = args.budget if args.budget is not None else BENCH_BUDGET

    corpus = generate_corpus(files, governing, seed=args.seed)
    gov_total = sum(1 for f in corpus if f.is_governing)
    print(f"[benchmark] real models via Ollama: {args.model} + nomic-embed-text\n"
          f"            {files} files, sending every file to the LLM then the "
          f"funnel; this takes several minutes on CPU\n")

    # Warm up so the first timed call is not penalised by model load.
    OllamaVerifier(model=args.model).verify(corpus[0])

    # Baseline: brute-force LLM over every non-media file with text.
    text_files = [f for f in corpus if not f.is_media and f.text]
    brute = OllamaVerifier(model=args.model)
    bar = _make_progress()
    t0 = time.perf_counter()
    brute_true = 0
    for i, f in enumerate(text_files, 1):
        if brute.verify(f) and f.is_governing:
            brute_true += 1
        bar("brute-force", i, len(text_files))
    brute_time = time.perf_counter() - t0

    # Funnel: cheap signals + graph rescue, LLM only on the head + audit.
    fv = OllamaVerifier(model=args.model)
    fe = OllamaEmbedder()
    t0 = time.perf_counter()
    result = run(corpus, max_llm_calls=budget, seed=args.seed, verifier=fv,
                 embedder=fe, progress=_make_progress())
    funnel_time = time.perf_counter() - t0
    funnel_true = sum(1 for f in result.surfaced if f.is_governing)

    speedup = brute_time / funnel_time if funnel_time else 0.0
    call_ratio = brute.calls / fv.calls if fv.calls else 0.0
    token_ratio = brute.tokens / fv.tokens if fv.tokens else 0.0

    def row(label, a, b):
        return f"{label:<24}: {a:>12} {b:>12}"

    report = "\n".join([
        "=" * 54,
        "  Timing: brute-force LLM vs the funnel",
        f"  {files} files, {gov_total} governing, model {args.model} (local CPU)",
        "=" * 54,
        f"{'':<24}  {'brute-force':>12} {'funnel':>12}",
        row("LLM calls", brute.calls, fv.calls),
        row("LLM tokens", f"{brute.tokens:,}", f"{fv.tokens:,}"),
        row("embedding calls", 0, fe.calls),
        row("wall-clock time", f"{brute_time:.1f}s", f"{funnel_time:.1f}s"),
        row("governing found", f"{brute_true}/{gov_total}", f"{funnel_true}/{gov_total}"),
        "-" * 54,
        f"funnel is {speedup:.1f}x faster, used {call_ratio:.0f}x fewer LLM calls "
        f"and {token_ratio:.0f}x fewer LLM tokens",
    ])
    print(report)

    metrics = {
        "mode": "benchmark", "model": args.model, "files": files,
        "governing_total": gov_total, "budget": budget, "seed": args.seed,
        "brute": {"llm_calls": brute.calls, "llm_tokens": brute.tokens,
                  "seconds": round(brute_time, 1), "governing_found": brute_true},
        "funnel": {"llm_calls": fv.calls, "llm_tokens": fv.tokens,
                   "embedding_calls": fe.calls, "seconds": round(funnel_time, 1),
                   "governing_found": funnel_true},
        "speedup": round(speedup, 1), "fewer_llm_calls": round(call_ratio, 1),
        "fewer_llm_tokens": round(token_ratio, 1),
    }
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"benchmark_{files}f_{args.model.replace(':', '-')}_{stamp}"
    path = _save(args.out, base, report, metrics)
    if path:
        print(f"\nsaved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", action="store_true", default=RUN_BENCHMARK,
                    help="time brute-force LLM over every file vs the funnel")
    ap.add_argument("--files", type=int, default=None)
    ap.add_argument("--governing", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None, help="max LLM calls (funnel)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--backend", choices=["mock", "ollama"], default=BACKEND,
                    help="'mock' = deterministic; 'ollama' = real local LLM")
    ap.add_argument("--model", default=MODEL, help="ollama model tag")
    ap.add_argument("--out", default=OUT_DIR,
                    help="folder for saved reports (empty string to skip saving)")
    args = ap.parse_args()

    if args.benchmark:
        run_benchmark(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
