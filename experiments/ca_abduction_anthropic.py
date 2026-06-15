"""CA abduction head-to-head for Anthropic reasoning models (adaptive thinking), done RIGHT.

The generic ca_scaling_llm.py infers LLM failure from unparseable text. That is unsafe for an
Anthropic reasoning model: with adaptive thinking the chain-of-thought counts against max_tokens,
so a fixed cap can truncate the thinking trace and return EMPTY text -- which looks identical to a
wrong answer but is not one. (o3 reasons server-side and never hits this; the comparison is only
fair if Opus is given room to finish.)

This runner records the ground truth instead of guessing it: per call it logs stop_reason and token
usage, flags `truncated = (stop_reason == "max_tokens")`, and a truncated call is EXCLUDED from the
capability stats and reported as its own count -- never scored as a failure. Budget is Opus's 128k
ceiling (it stops at end_turn far earlier on typical instances, ~27k tokens, so the ceiling only
costs more on a rare runaway). Instances, seeds, rule-training, and `ours` (the learned-MAP) match
ca_scaling_llm.py exactly, so this column is directly comparable to the cached o3 / Sonnet columns.

Resumable: each (model, rule, H, noise, idx) is appended and skipped on re-run.

Usage (needs ANTHROPIC_API_KEY in env; Opus abduction is ~$0.7-1.0/call):
    python experiments/ca_abduction_anthropic.py --model claude-opus-4-8 --effort medium \
        --n 5 --horizons 6 12 18 --noise 0.2 --dump results/ca_opus_abduction.jsonl
    python experiments/ca_abduction_anthropic.py --dump results/ca_opus_abduction.jsonl --aggregate-only
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
from pce.ca.headtohead import inverse_prompt, parse_inverse  # noqa: E402
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


def seed_for(rule, H, noise, idx):           # identical to ca_scaling_llm.py
    return (rule * 1_000_000) + (H * 10_000) + (int(round(noise * 100)) * 100) + idx


def load_done(path):
    done = {}
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["model"], r["rule"], r["H"], r["noise"], r["idx"])] = r
    return done


def _one_call(client, model, prompt, effort, max_tokens):
    with client.messages.stream(model=model, max_tokens=max_tokens,
                                thinking={"type": "adaptive"},
                                output_config={"effort": effort},
                                messages=[{"role": "user", "content": prompt}]) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return text, msg.stop_reason, msg.usage.input_tokens, msg.usage.output_tokens


# effort fallback chain: a runaway thinking trace that overruns the 128k ceiling at the requested
# effort is retried at successively lower effort (shorter trace) until it TERMINATES. 'low' is the
# floor and its trace is short enough to always fit, so every instance yields a real scored answer
# -- no instrumentation drops. The effort actually used is recorded per call.
_EFFORT_LADDER = ["max", "high", "medium", "low"]


def call_opus(client, model, prompt, effort, max_tokens):
    """Adaptive-thinking call that self-heals on truncation. Returns
    (text, stop_reason, in_tok, out_tok, effort_used, n_retries)."""
    start = _EFFORT_LADDER.index(effort) if effort in _EFFORT_LADDER else _EFFORT_LADDER.index("medium")
    ladder = _EFFORT_LADDER[start:] or ["low"]
    last = None
    for i, eff in enumerate(ladder):
        text, stop, itok, otok = _one_call(client, model, prompt, eff, max_tokens)
        last = (text, stop, itok, otok, eff, i)
        if stop != "max_tokens":
            return last
    return last  # even 'low' truncated (should never happen); caller flags it


def aggregate(path, L):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()] if Path(path).exists() else []
    if not rows:
        print("  (no results yet)"); return
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["H"], r["noise"], r["rule"])].append(r)
    models = sorted({r["model"] for r in rows})
    Hs = sorted({r["H"] for r in rows})
    print(f"\n  CA ABDUCTION (L={L}, exact joint x0&x1 recovery). LLM scored on COMPLETED calls only;")
    print(f"  truncated (stop=max_tokens, thinking overran the budget) is reported, never scored as fail.")
    print(f"  {'model':<22} {'H':>3} {'rule':<8} {'ours':>14} {'LLM(completed)':>16} {'trunc':>6} {'med_tok':>8}")
    for model in models:
        for H in Hs:
            for rule, name in RULES:
                rs = by.get((model, H, 0.2, rule)) or [v for k, v in by.items()
                                                       if k[0] == model and k[1] == H and k[3] == rule]
                if not rs:
                    continue
                n = len(rs)
                ok_o = sum(r["ours_exact"] for r in rs)
                done = [r for r in rs if not r.get("truncated")]
                m = len(done)
                ok_l = sum(r["llm_exact"] for r in done)
                trunc = n - m
                toks = sorted(r.get("out_tokens", 0) for r in rs)
                med = toks[len(toks) // 2] if toks else 0
                olo, ohi = wilson(ok_o, n)
                llo, lhi = wilson(ok_l, m)
                short = name.split()[1]
                print(f"  {model:<22} {H:>3} {short:<8} "
                      f"{f'{ok_o}/{n} [{olo:.2f},{ohi:.2f}]':>14} "
                      f"{f'{ok_l}/{m} [{llo:.2f},{lhi:.2f}]':>16} {f'{trunc}/{n}':>6} {med:>8}")
    print("\n  Read: compare LLM(completed) to ours per (rule,H). A high trunc count means the budget")
    print("  was too small for that cell -- raise --max-tokens and re-run those, don't read it as failure.")


def run(argv=None):
    ap = argparse.ArgumentParser(description="CA abduction for Anthropic reasoning models (truncation-aware)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="medium", help="adaptive-thinking effort: low|medium|high|max")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 18])
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128000, help="Opus ceiling; finishes earlier on typical instances")
    ap.add_argument("--dump", default="results/ca_opus_abduction.jsonl")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.dump, args.L)
        return

    import anthropic
    client = anthropic.Anthropic()
    model_key = f"anthropic:{args.model}"
    done = load_done(args.dump)
    print(f"CA abduction (truncation-aware) model={args.model} effort={args.effort} max_tokens={args.max_tokens}\n"
          f"  H={args.horizons} noise={args.noise} n={args.n}/rule  (resumable; {len(done)} cached)", flush=True)

    print("  training the learned rule models (CPU)...", flush=True)
    rule_models = {rule: LocalRuleModel(seed=0).fit(
        sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(1)), steps=1500)
        for rule, _ in RULES}
    denoisers = {}

    fh = open(args.dump, "a")
    for H in args.horizons:
        for rule, name in RULES:
            if (rule, H) not in denoisers:
                denoisers[(rule, H)] = TrajectoryDenoiser(rule_models[rule], args.L, H)
            den = denoisers[(rule, H)]
            for idx in range(args.n):
                key = (model_key, rule, H, args.noise, idx)
                if key in done:
                    continue
                rng = np.random.default_rng(seed_for(rule, H, args.noise, idx))
                x0, traj, obs = make_noisy_trajectory(rule, args.L, H, args.noise, rng)
                x1 = traj[1]
                est = den.denoise(obs)
                ours_exact = int(np.array_equal(est[0], x0) and np.array_equal(est[1], x1))
                try:
                    text, stop, itok, otok, eff_used, retries = call_opus(
                        client, args.model, inverse_prompt(rule, obs, args.noise),
                        args.effort, args.max_tokens)
                except Exception as e:
                    text, stop, itok, otok, eff_used, retries = f"<error: {e}>", "error", 0, 0, args.effort, 0
                truncated = int(stop == "max_tokens")            # only if even the lowest effort overran
                pred = parse_inverse(text, args.L) if not truncated else None
                llm_exact = int(pred is not None and np.array_equal(pred[0], x0)
                                and np.array_equal(pred[1], x1))
                rec = {"model": model_key, "rule": rule, "H": H, "noise": args.noise, "idx": idx,
                       "ours_exact": ours_exact, "llm_exact": llm_exact,
                       "llm_parsed": int(pred is not None), "truncated": truncated,
                       "stop_reason": stop, "effort_used": eff_used, "retries": retries,
                       "in_tokens": itok, "out_tokens": otok, "llm_text": text}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"  [{name} H={H} {idx+1}/{args.n}] ours={ours_exact} llm={llm_exact} "
                      f"parsed={int(pred is not None)} trunc={truncated} eff={eff_used} out_tok={otok}", flush=True)
    fh.close()
    aggregate(args.dump, args.L)


if __name__ == "__main__":
    run()
