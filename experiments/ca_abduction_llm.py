"""Experiment CA-013 (T3 head-to-head): the INVERSE problem (abduction).

The distinctive axis where the reducibility dial genuinely bites — and where no LLM
or NAR paper tests. Given a NOISY trajectory of a CA + the rule, recover the true
initial condition (x0, x1). This is abductive inference: find the IC whose forward
evolution best explains the noisy observations.

Our reasoner = the learned-dynamics MAP (`pce/ca/estimate.py:TrajectoryDenoiser`,
which enumerates ICs and rolls the LEARNED rule forward to match). The LLM is given
the rule + the noisy trajectory and asked the same question (code off).

Predictions (grounded in milestone (a), the inverted wall — sweet-spot noise ~0.2):
  * our reasoner recovers the IC, and BETTER on the IRREDUCIBLE Rule 110 than the
    reducible Rule 90 (chaos AIDS the inverse; reducibility is degenerate) — the
    FLIP relative to forward execution;
  * the LLM struggles with inverse inference for both (it cannot search ICs /
    reason backward) — abduction is a capability gap.

Usage (live):
    python experiments/ca_abduction_llm.py --model anthropic:claude-sonnet-4-6 --n 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.ca.env import RULE90, RULE110, sample_transitions  # noqa: E402
from pce.ca.estimate import TrajectoryDenoiser, make_noisy_trajectory  # noqa: E402
from pce.ca.headtohead import LLMInverseReasoner  # noqa: E402
from pce.ca.rule import LocalRuleModel  # noqa: E402


def _rule(rule, seed=0):
    return LocalRuleModel(seed=seed).fit(
        sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(seed + 1)), steps=1500)


def run(argv=None):
    p = argparse.ArgumentParser(description="CA-013: inverse head-to-head (T3)")
    p.add_argument("--model", default="anthropic:claude-sonnet-4-6")
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--H", type=int, default=12)
    p.add_argument("--noise", type=float, default=0.20)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump", default=None, help="JSONL path to cache every raw LLM transcript")
    p.add_argument("--max-tokens", type=int, default=3072,
                   help="raise for reasoning models so a 0 is real reasoning, not truncation")
    p.add_argument("--effort", default=None, help="openai reasoning_effort (minimal|low|medium|high)")
    args = p.parse_args(argv)

    from pce.llm import LLMClient
    provider, model = args.model.split(":", 1)
    llm = LLMInverseReasoner(LLMClient(provider, model, max_tokens=args.max_tokens,
                                       reasoning_effort=args.effort))
    dump = open(args.dump, "w") if args.dump else None

    print(f"CA-013  INVERSE head-to-head (abduction)  live LLM = {model}, code OFF\n"
          f"  recover (x0,x1) from a NOISY trajectory  L={args.L} H={args.H} noise={args.noise}\n")
    print(f"  {'rule':<24} {'ours bit-acc':>13} {'ours exact':>11} | "
          f"{'LLM bit-acc':>11} {'LLM exact':>10} {'parsed':>7}")
    for rule, name in [(RULE110, "Rule 110 (irreducible)"), (RULE90, "Rule 90 (reducible)")]:
        den = TrajectoryDenoiser(_rule(rule, args.seed), args.L, args.H)
        rng = np.random.default_rng(args.seed + 100)
        ob = oe = lb = le = parsed = 0.0
        for idx in range(args.n):
            x0, traj, obs = make_noisy_trajectory(rule, args.L, args.H, args.noise, rng)
            x1 = traj[1]
            est = den.denoise(obs)                    # MAP trajectory; IC = (est[0], est[1])
            ours_exact = int(np.array_equal(est[0], x0) and np.array_equal(est[1], x1))
            ob += ((est[0] == x0).mean() + (est[1] == x1).mean()) / 2
            oe += ours_exact
            try:
                pred, text = llm.solve(rule, obs, args.noise)
            except Exception as e:
                pred, text = None, f"<error: {e}>"
            llm_exact = 0
            if pred is not None:
                parsed += 1
                lb += ((pred[0] == x0).mean() + (pred[1] == x1).mean()) / 2
                llm_exact = int(np.array_equal(pred[0], x0) and np.array_equal(pred[1], x1))
                le += llm_exact
            if dump:
                dump.write(json.dumps({
                    "rule": rule, "name": name, "idx": idx,
                    "true_x0": "".join(map(str, x0.tolist())),
                    "true_x1": "".join(map(str, x1.tolist())),
                    "obs": ["".join(map(str, obs[t].tolist())) for t in range(obs.shape[0])],
                    "ours_x0": "".join(map(str, est[0].tolist())),
                    "ours_x1": "".join(map(str, est[1].tolist())),
                    "ours_exact": ours_exact,
                    "llm_parsed": int(pred is not None),  # 1 = produced valid FINAL_X0/X1 lines
                    "llm_exact": llm_exact,               # parsed-but-wrong => 1,0 => real reasoning miss
                    "llm_pred_x0": "".join(map(str, pred[0].tolist())) if pred is not None else None,
                    "llm_pred_x1": "".join(map(str, pred[1].tolist())) if pred is not None else None,
                    "llm_text": text,
                }) + "\n")
                dump.flush()
        n = args.n
        print(f"  {name:<24} {ob / n:>13.2f} {('%d/%d' % (oe, n)):>11} | "
              f"{(lb / max(parsed, 1)):>11.2f} {('%d/%d' % (le, n)):>10} {('%d/%d' % (parsed, n)):>7}")

    if dump:
        dump.close()
        print(f"\n  [transcripts cached -> {args.dump}]")

    print(f"\n  -> our reasoner does abduction (recover the IC), exploiting the inverted wall")
    print(f"     (irreducible Rule 110 is the MORE identifiable inverse — the flip vs forward).")
    print(f"     The LLM is asked the same question; report straight whether it can do inverse")
    print(f"     inference at all. (Small live slice — n={args.n}; the inverted-wall asymmetry")
    print(f"     itself is established exactly in ca_inverted_wall.py, not claimed from this slice.)")


if __name__ == "__main__":
    run()
