"""CA graded-metric companion to the exact-recovery abduction gap (CA-018).

Exact whole-IC recovery is the headline metric, but a reviewer can fairly ask
whether it *understates* the LLMs: are their misses NEAR-MISSES (a bit or two off,
which exact-match would unfairly score 0) or genuine COLLAPSE (far from the truth)?
This script answers that without any new API calls or retraining: it re-scores the
cached CA-018 transcripts (results/ca018.jsonl) under graded metrics.

For every cached (model, rule, H, idx) it regenerates the EXACT instance from the
same per-cell seed used by ca_scaling_llm.py (seed_for), so the true (x0,x1) is
reconstructed deterministically. It SELF-VALIDATES by recomputing our learned-MAP
ours_exact and asserting it equals the cached value on every row -- if all rows
match, the reconstruction is faithful and the LLM graded numbers are trustworthy.

Metrics:
  * Hamming-to-true-IC: bits wrong in (x0,x1) out of 2L (random-IC chance = L = 8).
  * near-miss rate: fraction of parsed answers within <=2 bits of the truth.
  * ours: mean Hamming, plus mean RANK of the true candidate under the learned
    likelihood (0 = MAP picks the truth = exact) and the oracle POSTERIOR MASS on
    the true IC (a noise-set difficulty indicator, from the exact enumerated posterior).

Usage:
    python experiments/ca_graded_metric.py --dump results/ca018.jsonl
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
from pce.ca.estimate import (  # noqa: E402
    TrajectoryDenoiser, enumerate_initials, make_noisy_trajectory,
    true_candidate_trajectories,
)
from pce.ca.headtohead import parse_inverse  # noqa: E402
from pce.ca.rule import LocalRuleModel  # noqa: E402

RULES = [(RULE110, "Rule 110"), (RULE90, "Rule 90")]


def seed_for(rule, H, noise, idx):
    """Identical to ca_scaling_llm.seed_for -- the per-instance stable seed."""
    return (rule * 1_000_000) + (H * 10_000) + (int(round(noise * 100)) * 100) + idx


def run(argv=None):
    ap = argparse.ArgumentParser(description="CA graded-metric re-scoring of CA-018")
    ap.add_argument("--dump", default="results/ca018.jsonl")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--out", default="results/ca_graded.jsonl")
    args = ap.parse_args(argv)
    L, noise = args.L, args.noise

    rows = [json.loads(x) for x in Path(args.dump).read_text().splitlines() if x.strip()]
    Hs = sorted({r["H"] for r in rows})
    W = (1 << np.arange(L)).astype(np.int64)
    beta = np.log((1 - noise) / noise)

    # rule models + per-(rule,H) precompute: learned & true candidate rollouts (obs-independent)
    rule_models = {rule: LocalRuleModel(seed=0).fit(
        sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(1)), steps=1500)
        for rule, _ in RULES}
    X0e, X1e = enumerate_initials(L)
    learned, truec = {}, {}
    for rule, _ in RULES:
        for H in Hs:
            learned[(rule, H)] = TrajectoryDenoiser(rule_models[rule], L, H).cand.reshape(2 ** (2 * L), -1)
            truec[(rule, H)] = true_candidate_trajectories(rule, X0e, X1e, H).reshape(2 ** (2 * L), -1)

    def true_index(x0, x1):
        return int(x0.dot(W)) * (2 ** L) + int(x1.dot(W))

    # per-instance graded record (ours), keyed by (rule,H,idx); validates ours_exact
    ours = {}
    mism = 0
    n_idx = max(r["idx"] for r in rows) + 1
    for rule, _ in RULES:
        for H in Hs:
            lc, tc = learned[(rule, H)], truec[(rule, H)]
            for idx in range(n_idx):
                rng = np.random.default_rng(seed_for(rule, H, noise, idx))
                x0, traj, obs = make_noisy_trajectory(rule, L, H, noise, rng)
                x1 = traj[1]
                of = obs.reshape(-1)
                ti = true_index(x0, x1)
                # ours (learned-MAP)
                lmatch = (lc == of[None]).sum(1)
                best = int(np.argmax(lmatch))
                ex0, ex1 = X0e[best], X1e[best]
                ours_exact = int(np.array_equal(ex0, x0) and np.array_equal(ex1, x1))
                ham = int((ex0 != x0).sum() + (ex1 != x1).sum())
                rank = int((lmatch > lmatch[ti]).sum())          # 0 == truth is the unique MAP
                # oracle posterior mass on the true IC (difficulty indicator, exact enumeration)
                lp = beta * (tc == of[None]).sum(1).astype(np.float64)
                lp -= lp.max()
                p = np.exp(lp); p /= p.sum()
                ours[(rule, H, idx)] = {"exact": ours_exact, "ham": ham, "rank": rank,
                                        "post": float(p[ti])}

    # attach LLM graded metrics from cached transcripts, validating ours_exact
    by = defaultdict(lambda: {"o": [], "l": defaultdict(list)})
    for r in rows:
        rule, H, idx, model = r["rule"], r["H"], r["idx"], r["model"]
        og = ours[(rule, H, idx)]
        if og["exact"] != r["ours_exact"]:
            mism += 1
        x0, traj, _ = make_noisy_trajectory(
            rule, L, H, noise, np.random.default_rng(seed_for(rule, H, noise, idx)))
        x1 = traj[1]
        by[(H,)]["o"].append(og)
        if r.get("llm_parsed"):
            pred = parse_inverse(r["llm_text"], L)
            if pred is not None:
                ham = int((pred[0] != x0).sum() + (pred[1] != x1).sum())
                by[(H,)]["l"][model].append(ham)

    assert mism == 0, f"reconstruction mismatch on {mism} rows -- instances not faithfully regenerated"
    print(f"  self-validation: ours_exact matches cache on ALL {len(rows)} rows (faithful reconstruction)\n")

    out = open(args.out, "w")
    print(f"  GRADED abduction metrics (L={L}, noise={noise}; random-IC Hamming chance = {L}/{2*L} bits)")
    print(f"  {'H':>3} | {'ours exact':>10} {'ours Ham':>9} {'ours rank':>10} {'oracle post':>12} |"
          f" {'model':<26} {'parsed':>6} {'LLM Ham':>8} {'near-miss<=2':>12}")
    for H in Hs:
        o = by[(H,)]["o"]
        oe = np.mean([d["exact"] for d in o]); oh = np.mean([d["ham"] for d in o])
        orank = np.mean([d["rank"] for d in o]); opost = np.mean([d["post"] for d in o])
        first = True
        for model in sorted(by[(H,)]["l"]):
            hams = by[(H,)]["l"][model]
            nm = np.mean([h <= 2 for h in hams]) if hams else float("nan")
            mh = np.mean(hams) if hams else float("nan")
            head = (f"  {H:>3} | {oe:>10.3f} {oh:>9.2f} {orank:>10.1f} {opost:>12.3f} |"
                    if first else f"  {'':>3} | {'':>10} {'':>9} {'':>10} {'':>12} |")
            print(f"{head} {model:<26} {len(hams):>6} {mh:>8.2f} {nm:>12.2f}")
            out.write(json.dumps({"H": H, "model": model, "ours_exact": float(oe),
                                  "ours_ham": float(oh), "ours_rank": float(orank),
                                  "oracle_post": float(opost), "llm_parsed": len(hams),
                                  "llm_ham": float(mh), "llm_nearmiss_le2": float(nm)}) + "\n")
            first = False
    out.close()
    print(f"\n  [graded metrics -> {args.out}]")
    print("  Reading: ours Hamming ~ 0 and rank ~ 0 (tracks the oracle); if LLM Hamming ~ L the misses")
    print("  are COLLAPSE not near-miss, so exact-match does not understate the gap.")


if __name__ == "__main__":
    run()
