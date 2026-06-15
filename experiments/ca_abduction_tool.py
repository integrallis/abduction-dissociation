"""Tool-augmented LLM abduction baseline: Opus 4.8 WITH server-side code execution.

The paper's main LLM baselines are tool-free on purpose -- letting the model write a solver turns
abduction into retrieval of a known algorithm rather than in-context inverse reasoning. This runner
is the complementary *practical* baseline the reviewer asked for: the same Opus 4.8, same seeded
instances, but allowed to write and run Python (Anthropic's server-side code_execution tool). It tests
whether the abduction gap is about exact inverse search specifically -- if so, a model that can code
the enumerate-and-rollout MAP should close most of it.

Records per call whether code actually ran, plus stop_reason and tokens. Same instances/seeds and the
learned-MAP `ours` reference as ca_abduction_anthropic.py, so it drops in alongside that column.

Usage (needs ANTHROPIC_API_KEY; ~$0.05/call -- code execution offloads the compute):
    python experiments/ca_abduction_tool.py --n 5 --horizons 6 12 18 --noise 0.2 \
        --dump results/ca_opus_tool.jsonl
    python experiments/ca_abduction_tool.py --aggregate-only --dump results/ca_opus_tool.jsonl
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
CODE_TOOL = [{"type": "code_execution_20260120", "name": "code_execution"}]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def seed_for(rule, H, noise, idx):           # identical to ca_scaling_llm.py / ca_abduction_anthropic.py
    return (rule * 1_000_000) + (H * 10_000) + (int(round(noise * 100)) * 100) + idx


def tool_prompt(rule, obs, noise):
    return inverse_prompt(rule, obs, noise).replace(
        "Do not use any tools or code; reason it out yourself.",
        "You MAY write and run Python code to solve this. End with the two FINAL_ lines.")


def call_opus_tool(client, model, prompt, effort, max_tokens, max_turns=8):
    """Opus + server-side code execution, resolving pause_turn. Returns
    (text, stop_reason, in_tok, out_tok, ran_code)."""
    msgs = [{"role": "user", "content": prompt}]
    itok = otok = 0
    ran = False
    stop = "error"
    for _ in range(max_turns):
        r = client.messages.create(model=model, max_tokens=max_tokens,
                                   thinking={"type": "adaptive"},
                                   output_config={"effort": effort},
                                   tools=CODE_TOOL, messages=msgs)
        itok += r.usage.input_tokens
        otok += r.usage.output_tokens
        ran = ran or any(getattr(b, "type", "") == "server_tool_use" for b in r.content)
        stop = r.stop_reason
        if stop == "pause_turn":            # server-side tool loop hit its cap; continue
            msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": r.content}]
            continue
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return text, stop, itok, otok, ran
    return text, "pause_turn", itok, otok, ran


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
        by[(r["H"], r["rule"])].append(r)
    print(f"\n  TOOL-AUGMENTED CA abduction (Opus 4.8 + code execution), L={L}, exact joint recovery:")
    print(f"  {'H':>3} {'rule':<8} {'ours':>10} {'Opus+code':>12} {'ran_code':>9}")
    for H in sorted({r["H"] for r in rows}):
        for rule, name in RULES:
            rs = by.get((H, rule))
            if not rs:
                continue
            n = len(rs); o = sum(r["ours_exact"] for r in rs)
            l = sum(r["llm_exact"] for r in rs); rc = sum(r.get("ran_code", 0) for r in rs)
            short = name.split()[1]
            print(f"  {H:>3} {short:<8} {f'{o}/{n}':>10} {f'{l}/{n}':>12} {f'{rc}/{n}':>9}")
    tot = len(rows); ok = sum(r["llm_exact"] for r in rows); oo = sum(r["ours_exact"] for r in rows)
    lo, hi = wilson(ok, tot)
    print(f"\n  POOLED: ours {oo}/{tot}, Opus+code {ok}/{tot} [{lo:.2f},{hi:.2f}], "
          f"ran_code {sum(r.get('ran_code',0) for r in rows)}/{tot}")
    print("  Read: with code allowed, the LLM writes the enumerate-and-rollout MAP and the abduction")
    print("  gap largely CLOSES -- the missing capability is exact inverse search, not 'reasoning'.")


def run(argv=None):
    ap = argparse.ArgumentParser(description="Tool-augmented CA abduction (Opus 4.8 + code execution)")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--horizons", type=int, nargs="+", default=[6, 12, 18])
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--dump", default="results/ca_opus_tool.jsonl")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args.dump, args.L); return

    import anthropic
    client = anthropic.Anthropic()
    model_key = f"anthropic:{args.model}+code"
    done = load_done(args.dump)
    print(f"Tool-augmented CA abduction: {args.model} + code execution  effort={args.effort}\n"
          f"  H={args.horizons} noise={args.noise} n={args.n}/rule (resumable; {len(done)} cached)", flush=True)

    print("  training the learned rule models (CPU, for the ours reference)...", flush=True)
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
                    text, stop, itok, otok, ran = call_opus_tool(
                        client, args.model, tool_prompt(rule, obs, args.noise), args.effort, args.max_tokens)
                except Exception as e:
                    text, stop, itok, otok, ran = f"<error: {e}>", "error", 0, 0, False
                pred = parse_inverse(text, args.L)
                llm_exact = int(pred is not None and np.array_equal(pred[0], x0)
                                and np.array_equal(pred[1], x1))
                rec = {"model": model_key, "rule": rule, "H": H, "noise": args.noise, "idx": idx,
                       "ours_exact": ours_exact, "llm_exact": llm_exact,
                       "llm_parsed": int(pred is not None), "ran_code": int(ran),
                       "stop_reason": stop, "in_tokens": itok, "out_tokens": otok, "llm_text": text}
                fh.write(json.dumps(rec) + "\n"); fh.flush()
                print(f"  [{name} H={H} {idx+1}/{args.n}] ours={ours_exact} llm={llm_exact} "
                      f"parsed={int(pred is not None)} ran_code={int(ran)} out_tok={otok}", flush=True)
    fh.close()
    aggregate(args.dump, args.L)


if __name__ == "__main__":
    run()
