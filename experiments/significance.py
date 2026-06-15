"""Paired significance tests on the shared seeded instances (McNemar's exact test).

The abduction head-to-heads reuse stable per-instance seeds, so ours and each LLM are scored on the
SAME instances -- a paired design. Wilson intervals (used in the paper) ignore that pairing; the exact
McNemar test uses it. For each LLM we tabulate the discordant pairs on matched instances:
  b = ours correct, LLM wrong ;  c = ours wrong, LLM correct
and report the two-sided exact-binomial (McNemar) p-value, p = min(1, 2*sum_{i=0}^{min(b,c)} C(n,i) 0.5^n),
n = b+c. Pure analysis of the cached transcripts -- no API access, deterministic.

Usage:
    python experiments/significance.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

CACHES = {
    "o3 (n=15/cell)": ("results/ca018.jsonl", "openai:o3"),
    "Sonnet 4.6 (n=15/cell)": ("results/ca018.jsonl", "anthropic:claude-sonnet-4-6"),
    "Opus 4.8 (n=5/cell)": ("results/ca_opus_abduction.jsonl", "anthropic:claude-opus-4-8"),
}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact-binomial (McNemar) p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


def load(path, model):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    # Opus runs exclude truncated calls from scoring; honor that here too.
    return [r for r in rows if r["model"] == model and not r.get("truncated")]


def main():
    print("Paired McNemar exact test, ours vs each LLM on shared seeded CA-abduction instances\n")
    print(f"  {'comparison':<26} {'n':>4} {'ours':>6} {'llm':>6} {'b':>4} {'c':>4} {'p (two-sided)':>16}")
    for name, (path, model) in CACHES.items():
        if not Path(path).exists():
            print(f"  {name:<26} (missing {path})"); continue
        rows = load(path, model)
        n = len(rows)
        ours = sum(r["ours_exact"] for r in rows)
        llm = sum(r["llm_exact"] for r in rows)
        b = sum(1 for r in rows if r["ours_exact"] and not r["llm_exact"])
        c = sum(1 for r in rows if not r["ours_exact"] and r["llm_exact"])
        p = mcnemar_exact_p(b, c)
        ps = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
        print(f"  {name:<26} {n:>4} {ours:>6} {llm:>6} {b:>4} {c:>4} {ps:>16}")
    print("\n  b = ours-right/LLM-wrong, c = ours-wrong/LLM-right. A tiny p rejects 'ours and the LLM")
    print("  are equally likely to win a discordant instance', i.e. ours is paired-significantly better.")


if __name__ == "__main__":
    main()
