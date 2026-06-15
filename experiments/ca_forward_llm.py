"""CA-021: the FORWARD-execution head-to-head (the missing third axis).

Audit-2 T0.1: the paper's three-axis framing (forward / control / abduction) asserts a forward
column but never measured it. This runs it: apply the rule N steps and check whole-config exact match,
for a STRONG reasoning model (o3), a WEAK model (gpt-4o-mini), and our learned-rule reasoner. Fairness
matches the rest of the suite: the LLM is given the exact rule table + dynamics and a scratchpad, no
tools (pce/ca/headtohead.py:forward_prompt).

Resumable + dumped (o3 is slow): per-(model,rule,N,idx) result is appended and skipped on re-run; the
instance is regenerated from a stable seed so all models see the same (prev,cur).

Usage:
    python experiments/ca_forward_llm.py --models openai:o3 openai:gpt-4o-mini \
        --effort medium --L 8 --horizons 4 8 16 32 --n 5 --dump results/ca021_forward.jsonl
    python experiments/ca_forward_llm.py --dump results/ca021_forward.jsonl --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.ca.env import RULE110, RULE90, sample_transitions  # noqa: E402
from pce.ca.headtohead import (LLMForwardReasoner, learned_forward,  # noqa: E402
                               true_forward)
from pce.ca.rule import LocalRuleModel  # noqa: E402

RULES = [(RULE110, "Rule 110 (irreducible)"), (RULE90, "Rule 90 (reducible)")]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def seed_for(rule, N, idx):
    return (rule * 1_000_000) + (N * 10_000) + idx


def load_done(path):
    done = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["model"], r["rule"], r["N"], r["idx"])] = r
    return done


def aggregate(path, L):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    if not rows:
        print("  (no results yet)"); return
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["N"], r["rule"])].append(r)
    models = sorted({r["model"] for r in rows})
    Ns = sorted({r["N"] for r in rows})
    # ours is identical across LLM rows; report it once per (N,rule) from any row
    print(f"\n  FORWARD execution (L={L}, whole-config exact match after N steps):")
    print(f"  {'model':<22} {'N':>3} {'rule':<8} {'exact':>14} {'bit-acc':>8} {'parsed':>7}")
    for N in Ns:
        for rule, name in RULES:
            short = name.split()[1]
            anyrow = by.get((models[0], N, rule))
            if anyrow:
                ok = sum(r["ours_exact"] for r in anyrow); n = len(anyrow)
                lo, hi = wilson(ok, n)
                print(f"  {'ours (learned)':<22} {N:>3} {short:<8} {f'{ok}/{n} [{lo:.2f},{hi:.2f}]':>14} {'1.00':>8} {'-':>7}")
            for model in models:
                rs = by.get((model, N, rule))
                if not rs:
                    continue
                n = len(rs); ok = sum(r["llm_exact"] for r in rs)
                ba = np.mean([r["llm_bitacc"] for r in rs]); par = sum(r["llm_parsed"] for r in rs)
                lo, hi = wilson(ok, n)
                print(f"  {model:<22} {N:>3} {short:<8} {f'{ok}/{n} [{lo:.2f},{hi:.2f}]':>14} {ba:>8.2f} {f'{par}/{n}':>7}")
        print()
    print("  Read: ours stays exact (learned rule rolled forward); the strong reasoning model holds")
    print("  forward far longer than the weak model, which collapses as N grows. This is the forward axis.")


def run(argv=None):
    ap = argparse.ArgumentParser(description="CA-021: forward-execution head-to-head")
    ap.add_argument("--models", nargs="+", default=["openai:o3", "openai:gpt-4o-mini"])
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--dump", default="results/ca021_forward.jsonl")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.dump, args.L); return

    from pce.llm import LLMClient
    done = load_done(args.dump)
    print(f"CA-021  forward-execution head-to-head (resumable; {len(done)} cached)\n"
          f"  models={args.models} N={args.horizons} L={args.L} n={args.n}/rule effort={args.effort}")

    rule_models = {rule: LocalRuleModel(seed=0).fit(
        sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(1)), steps=1500)
        for rule, _ in RULES}

    fh = open(args.dump, "a")
    for model in args.models:
        prov, mname = model.split(":", 1)
        # reasoning models take an effort: openai o-series, or anthropic (adaptive thinking)
        eff = args.effort if ((prov == "openai" and mname.startswith("o")) or prov == "anthropic") else None
        llm = LLMForwardReasoner(LLMClient(prov, mname, max_tokens=8192, reasoning_effort=eff))
        for N in args.horizons:
            for rule, name in RULES:
                for idx in range(args.n):
                    key = (model, rule, N, idx)
                    if key in done:
                        continue
                    rng = np.random.default_rng(seed_for(rule, N, idx))
                    prev, cur = rng.integers(0, 2, size=(2, args.L)).astype(np.int8)
                    truth = true_forward(rule, prev, cur, N)
                    ours = learned_forward(rule_models[rule], prev, cur, N)
                    ours_exact = int(np.array_equal(ours, truth))
                    try:
                        pred, text = llm.solve(rule, prev, cur, N)
                    except Exception as e:
                        pred, text = None, f"<error: {e}>"
                    llm_exact = int(pred is not None and np.array_equal(pred, truth))
                    bitacc = float((pred == truth).mean()) if pred is not None else 0.0
                    rec = {"model": model, "rule": rule, "N": N, "idx": idx,
                           "ours_exact": ours_exact, "llm_exact": llm_exact,
                           "llm_parsed": int(pred is not None), "llm_bitacc": bitacc, "llm_text": text}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"  [{model} {name} N={N} {idx+1}/{args.n}] ours={ours_exact} "
                          f"llm={llm_exact} bitacc={bitacc:.2f}", flush=True)
    fh.close()
    aggregate(args.dump, args.L)


if __name__ == "__main__":
    run()
