"""CA-018 (S2 scaling, paid LLM side): abduction gap at scale + 'widens with difficulty'.

The paid half of S2. Establishes the LLM side of the abduction gap with multi-instance CIs, and the
'widens with difficulty' claim along the axis where it actually holds: HORIZON H (more frames to
hand-denoise hurts the LLM; more frames give our learned-MAP more constraints, so ours holds at the
recoverability ceiling). Noise is NOT that axis — it degrades both (the oracle ceiling falls; see
CA-017) — so difficulty here = H in {6,12,18} at fixed noise.

RESUMABLE by design: every (model, rule, H, noise, idx) result is appended to the dump and skipped on
re-run, and the instance is generated from a STABLE per-cell seed so ours and the LLM see the same
trajectory across runs. o3 is slow (~3 min/call) -> this spans hours; kill/relaunch freely, it resumes.
Re-aggregate the (even partial) dump anytime with --aggregate-only.

Usage:
    python experiments/ca_scaling_llm.py --models openai:o3 anthropic:claude-sonnet-4-6 --effort medium \
        --n 15 --horizons 6 12 18 --noise 0.2 --dump results/ca018.jsonl
    python experiments/ca_scaling_llm.py --dump results/ca018.jsonl --aggregate-only
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
from pce.ca.estimate import TrajectoryDenoiser, make_noisy_trajectory  # noqa: E402
from pce.ca.headtohead import LLMInverseReasoner  # noqa: E402
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


def seed_for(rule, H, noise, idx):
    return (rule * 1_000_000) + (H * 10_000) + (int(round(noise * 100)) * 100) + idx


def load_done(path):
    done = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["model"], r["rule"], r["H"], r["noise"], r["idx"])] = r
    return done


def aggregate(path, L):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    if not rows:
        print("  (no results yet)"); return
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["H"], r["noise"], r["rule"])].append(r)
    models = sorted({r["model"] for r in rows})
    Hs = sorted({r["H"] for r in rows})
    print(f"\n  ABDUCTION GAP at scale (L={L}, exact joint x0&x1 recovery; ours=learned-MAP):")
    print(f"  {'model':<26} {'H':>3} {'rule':<10} {'ours':>14} {'LLM':>16} {'parsed':>7} {'gap':>7}")
    for model in models:
        for H in Hs:
            for rule, name in RULES:
                rs = by.get((model, H, 0.2, rule)) or [v for k, v in by.items()
                                                        if k[0] == model and k[1] == H and k[3] == rule]
                rs = [r for r in (rs or [])]
                if not rs:
                    continue
                n = len(rs)
                ok_o = sum(r["ours_exact"] for r in rs)
                ok_l = sum(r["llm_exact"] for r in rs)
                par = sum(r["llm_parsed"] for r in rs)
                olo, ohi = wilson(ok_o, n); llo, lhi = wilson(ok_l, n)
                short = name.split()[1]
                print(f"  {model:<26} {H:>3} {short:<10} "
                      f"{f'{ok_o}/{n} [{olo:.2f},{ohi:.2f}]':>14} "
                      f"{f'{ok_l}/{n} [{llo:.2f},{lhi:.2f}]':>16} {f'{par}/{n}':>7} "
                      f"{(ok_o-ok_l)/n:>+7.2f}")
    print("\n  'widens with difficulty' = the ours-minus-LLM gap should GROW with H (ours holds at the")
    print("  recoverability ceiling; the LLM degrades as more frames must be hand-denoised).")


def run(argv=None):
    ap = argparse.ArgumentParser(description="CA-018: S2 paid LLM-at-scale abduction gap")
    ap.add_argument("--models", nargs="+", default=["openai:o3", "anthropic:claude-sonnet-4-6"])
    ap.add_argument("--effort", default="medium", help="openai reasoning_effort")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 18])
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--dump", default="results/ca018.jsonl")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.dump, args.L)
        return

    from pce.llm import LLMClient
    done = load_done(args.dump)
    n_done = len(done)
    print(f"CA-018  paid LLM-at-scale abduction gap (resumable; {n_done} results already cached)\n"
          f"  models={args.models} H={args.horizons} noise={args.noise} n={args.n}/rule, effort={args.effort}")

    # cache rule models + denoisers (denoiser depends on (rule,H))
    rule_models = {rule: LocalRuleModel(seed=0).fit(
        sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(1)), steps=1500)
        for rule, _ in RULES}
    denoisers = {}

    fh = open(args.dump, "a")
    for model in args.models:
        prov, mname = model.split(":", 1)
        eff = args.effort if prov in ("openai", "anthropic") else None
        llm = LLMInverseReasoner(LLMClient(prov, mname, max_tokens=8192, reasoning_effort=eff))
        for H in args.horizons:
            for rule, name in RULES:
                if (rule, H) not in denoisers:
                    denoisers[(rule, H)] = TrajectoryDenoiser(rule_models[rule], args.L, H)
                den = denoisers[(rule, H)]
                for idx in range(args.n):
                    key = (model, rule, H, args.noise, idx)
                    if key in done:
                        continue
                    rng = np.random.default_rng(seed_for(rule, H, args.noise, idx))
                    x0, traj, obs = make_noisy_trajectory(rule, args.L, H, args.noise, rng)
                    x1 = traj[1]
                    est = den.denoise(obs)
                    ours_exact = int(np.array_equal(est[0], x0) and np.array_equal(est[1], x1))
                    try:
                        pred, text = llm.solve(rule, obs, args.noise)
                    except Exception as e:
                        pred, text = None, f"<error: {e}>"
                    llm_exact = int(pred is not None and np.array_equal(pred[0], x0)
                                    and np.array_equal(pred[1], x1))
                    rec = {"model": model, "rule": rule, "H": H, "noise": args.noise, "idx": idx,
                           "ours_exact": ours_exact, "llm_exact": llm_exact,
                           "llm_parsed": int(pred is not None), "llm_text": text}
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                    print(f"  [{model} {name} H={H} {idx+1}/{args.n}] ours={ours_exact} llm={llm_exact} "
                          f"parsed={int(pred is not None)}", flush=True)
    fh.close()
    aggregate(args.dump, args.L)


if __name__ == "__main__":
    run()
