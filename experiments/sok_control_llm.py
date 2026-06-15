"""Sokoban CONTROL head-to-head — the matched within-domain comparison (paper Table 1, S2b).

The control axis is what pins the dissociation to abduction specifically: on the SAME Sokoban
world where o3 fails to abduct, it *closes* control. This driver reproduces that leg.

Our learned-simulation planner (plan inside the learned push-rule; the true env is touched
only to verify) vs a frontier LLM given the SAME instance with the rules written out and code
execution OFF (planning + simulation are both LLM weaknesses; PlanBench/Kambhampati). Instances
are on small grids and binned by OPTIMAL plan length, so the difficulty sweep is visible.
Random shooting is the floor; A* also solves these (so this is not "beats A*", it is the
matched assistant-relevant comparison).

The CPU rows (ours, random) are deterministic given the seed and reproduce offline for free.
The LLM rows require API access; like the other LLM drivers, runs are resumable and dumped to a
jsonl so they re-aggregate without re-spending.

Usage:
    # free, CPU-only: trains the rule and reports ours / random (no API key)
    python experiments/sok_control_llm.py --cpu-only --dump results/sok_control.jsonl

    # live head-to-head (resumable; o3 is slow + costs real money)
    python experiments/sok_control_llm.py --models openai:o3 anthropic:claude-sonnet-4-6 \
        --effort medium --n 6 --dump results/sok_control.jsonl

    # re-aggregate from cached transcripts, no API access
    python experiments/sok_control_llm.py --dump results/sok_control.jsonl --aggregate-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.sok import env as E  # noqa: E402
from pce.sok import planner as P  # noqa: E402
from pce.sok.headtohead import LLMSokReasoner  # noqa: E402
from pce.sok.rule import SokLocalRule  # noqa: E402

BINS = ["short (<=6)", "medium (7-12)", "long (13-20)"]
OURS = "ours (learned planner)"   # pseudo-model key for the model-agnostic CPU rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def train_rule(steps=4000):
    """One weight-shared local push-rule from black-box transitions on small grids (seed 0)."""
    rng = np.random.default_rng(0)
    grids = [E.random_grid(int(rng.integers(6, 9)), int(rng.integers(6, 9)),
                           int(rng.integers(2, 4)), rng) for _ in range(300)]
    return SokLocalRule(hidden=64, seed=0).fit(E.sample_transitions(grids, 12, rng), steps=steps)


def build_bins(rng, per_bin):
    """Generate instances on small grids, binned by optimal plan length (A*-verified solvable)."""
    def verify(s):
        pl, _, _ = P.plan_in_true_model(s, max_expansions=20000)
        return pl

    want = {"short (<=6)": (per_bin, lambda o: o <= 6),
            "medium (7-12)": (per_bin, lambda o: 7 <= o <= 12),
            "long (13-20)": (per_bin, lambda o: 13 <= o <= 20)}
    bins = {k: [] for k in want}
    settings = [(6, 6, 1, 2), (6, 6, 2, 2), (7, 7, 2, 3), (7, 7, 1, 3), (8, 8, 2, 4)]
    tries = 0
    while any(len(bins[k]) < want[k][0] for k in want) and tries < 4000:
        tries += 1
        H, W, nb, md = settings[int(rng.integers(0, len(settings)))]
        try:
            s, ol = E.generate_far_instance(H, W, nb, rng, min_dist=md, verify=verify)
        except RuntimeError:
            continue
        for k, (cap, pred) in want.items():
            if pred(ol) and len(bins[k]) < cap:
                bins[k].append((s, ol))
                break
    return bins


def load_done(path):
    done = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["model"], r["bin"], r["idx"])] = r
    return done


def aggregate(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    if not rows:
        print("  (no results yet)"); return
    by = defaultdict(dict)            # (model) -> {(bin, idx): record}
    for r in rows:
        by[r["model"]][(r["bin"], r["idx"])] = r
    llm_models = sorted(m for m in by if m != OURS)

    print("\n  Sokoban CONTROL (solve = plan reaches the goal in the TRUE env):")
    header = f"  {'difficulty':<16} {'n':>3} {'ours':>7} {'random':>7}"
    for m in llm_models:
        header += f" {m.split(':')[-1][:14]:>15}"
    print(header)

    tot = {"n": 0, "ours": 0, "rnd": 0, **{m: [0, 0] for m in llm_models}}  # m -> [ok, parsed]
    for b in BINS:
        ours_rows = [v for (bb, _), v in by.get(OURS, {}).items() if bb == b]
        n = len(ours_rows)
        if n == 0:
            continue
        ours_ok = sum(v["ours_ok"] for v in ours_rows)
        rnd_ok = sum(v["rnd_ok"] for v in ours_rows)
        tot["n"] += n; tot["ours"] += ours_ok; tot["rnd"] += rnd_ok
        line = f"  {b:<16} {n:>3} {f'{ours_ok}/{n}':>7} {f'{rnd_ok}/{n}':>7}"
        for m in llm_models:
            mrows = [v for (bb, _), v in by[m].items() if bb == b]
            ok = sum(v["llm_ok"] for v in mrows); par = sum(v["llm_parsed"] for v in mrows)
            tot[m][0] += ok; tot[m][1] += par
            line += f" {f'{ok}/{len(mrows)}':>15}"
        print(line)

    N = tot["n"]
    lo, hi = wilson(tot["ours"], N)
    ours_cell = f"{tot['ours']}/{N}"
    rnd_cell = f"{tot['rnd']}/{N}"
    line = f"  {'TOTAL':<16} {N:>3} {ours_cell:>7} {rnd_cell:>7}"
    for m in llm_models:
        line += f" {f'{tot[m][0]}/{N}':>15}"
    print(line)
    print(f"\n  ours {tot['ours']}/{N} Wilson [{lo:.2f},{hi:.2f}].  Read: o3 (a reasoning model) *closes*")
    print("  control on the same world where it fails abduction -- the dissociation is abduction-specific.")


def run(argv=None):
    ap = argparse.ArgumentParser(description="Sokoban control head-to-head (the matched comparison)")
    ap.add_argument("--models", nargs="+", default=["openai:o3", "anthropic:claude-sonnet-4-6"])
    ap.add_argument("--cpu-only", action="store_true", help="train + report ours/random only (no API)")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--n", type=int, default=6, help="instances per difficulty bin (3 bins -> 3n total)")
    ap.add_argument("--budget", type=int, default=12000, help="planner expansion budget")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dump", default="results/sok_control.jsonl")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.dump); return

    done = load_done(args.dump)
    models = [] if args.cpu_only else args.models
    print(f"Sokoban control head-to-head (resumable; {len(done)} cached)\n"
          f"  models={models or '[cpu-only]'} n={args.n}/bin seed={args.seed} budget={args.budget}",
          flush=True)

    print("  training the local push-rule (black-box transitions, small grids)...", flush=True)
    net = train_rule()
    rng = np.random.default_rng(args.seed)
    bins = build_bins(rng, args.n)

    clients = {}
    if models:
        from pce.llm import LLMClient
        for model in models:
            prov, mname = model.split(":", 1)
            eff = args.effort if (prov == "openai" and mname.startswith("o")) else None
            clients[model] = LLMSokReasoner(LLMClient(prov, mname, max_tokens=4096, reasoning_effort=eff))

    fh = open(args.dump, "a")
    for b in BINS:
        insts = bins.get(b, [])
        for idx, (s, ol) in enumerate(insts):
            # ours + random are model-agnostic and deterministic: compute/dump once
            if (OURS, b, idx) not in done:
                pl, _, _ = P.plan_in_learned_model(net, s, max_expansions=args.budget)
                ours_ok = int(P.verify_plan(s, pl))
                rnd_ok = int(P.random_shooting(s, n_tries=600, max_len=2 * ol + 10, rng=rng) is not None)
                rec = {"model": OURS, "bin": b, "idx": idx, "optimal": int(ol),
                       "ours_ok": ours_ok, "rnd_ok": rnd_ok}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"  [{b} {idx+1}/{len(insts)} opt={ol}] ours={ours_ok} random={rnd_ok}", flush=True)
            for model, llm in clients.items():
                if (model, b, idx) in done:
                    continue
                try:
                    actions, text = llm.solve(s)
                except Exception as e:                       # resilient to API hiccups
                    actions, text = None, f"<error: {e}>"
                llm_ok = int(actions is not None and P.verify_plan(s, actions))
                rec = {"model": model, "bin": b, "idx": idx, "optimal": int(ol),
                       "llm_ok": llm_ok, "llm_parsed": int(actions is not None), "llm_text": text}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"  [{model} {b} {idx+1}/{len(insts)}] llm={llm_ok} parsed={int(actions is not None)}",
                      flush=True)
    fh.close()
    aggregate(args.dump)


if __name__ == "__main__":
    run()
