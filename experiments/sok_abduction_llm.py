"""Experiment SOK-004 (S3, abduction) — recover the initial layout from a NOISY trajectory.

The DISTINCTIVE test (on the CA, abduction was the mode where even o3 scored 0/6). A
trajectory s_0..s_T is produced by a KNOWN action sequence; every snapshot's box/agent cells
are independently bit-flipped with probability `noise` (walls/goals clean). Recover the true
initial board s_0.

  - ours: MAP over candidate initial boards rolled forward in the LEARNED model (the dynamics
    aggregate evidence across all noisy frames — pce/sok/abduce.py).
  - oracle: the same MAP with the TRUE dynamics (recoverability upper bound).
  - raw: denoise the first frame alone (no dynamics) — the floor.
  - degraded: the MAP with an under-trained model (load-bearing gate: it should collapse).
  - LLM: given the noisy trajectory (ASCII) + actions + rules, code OFF, recover s_0.

Irreversibility is the twist vs the (reversible) CA: forward push-dynamics destroy
information, so the inverse is non-trivial. Exact recovery = the BOX placement + AGENT cell.

Usage (live):
    python experiments/sok_abduction_llm.py --model anthropic:claude-sonnet-4-6 --n 8
    python experiments/sok_abduction_llm.py --model openai:o3 --n 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.sok import abduce as A  # noqa: E402
from pce.sok import env as E  # noqa: E402
from pce.sok.headtohead import LLMSokAbducer  # noqa: E402
from pce.sok.rule import SokLocalRule  # noqa: E402


def train_rule(steps):
    rng = np.random.default_rng(0)
    grids = [E.random_grid(int(rng.integers(6, 9)), int(rng.integers(6, 9)),
                           int(rng.integers(2, 4)), rng) for _ in range(300)]
    return SokLocalRule(hidden=64, seed=0).fit(E.sample_transitions(grids, 12, rng), steps=steps)


def run(argv=None):
    p = argparse.ArgumentParser(description="SOK-004: abduction head-to-head (S3)")
    p.add_argument("--model", default="anthropic:claude-sonnet-4-6")
    p.add_argument("--H", type=int, default=6)
    p.add_argument("--W", type=int, default=6)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--T", type=int, default=6)
    p.add_argument("--noise", type=float, default=0.12)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seed", type=int, default=100)
    args = p.parse_args(argv)

    from pce.llm import LLMClient
    provider, model = args.model.split(":", 1)
    llm = LLMSokAbducer(LLMClient(provider, model, max_tokens=4096))

    print(f"SOK-004  abduction head-to-head (recover s0 from a NOISY trajectory)  "
          f"LLM={model}, code OFF", flush=True)
    print(f"  {args.H}x{args.W}, {args.k} boxes, T={args.T}, noise={args.noise}, n={args.n}\n",
          flush=True)
    full = train_rule(4000)
    degraded = train_rule(150)
    rng = np.random.default_rng(args.seed)

    ours = orac = raw = deg = llm_ok = parsed = 0
    for _ in range(args.n):
        s0 = E.random_grid(args.H, args.W, args.k, rng)
        acts = [int(rng.integers(0, 4)) for _ in range(args.T)]
        _, obs = A.make_noisy_trajectory(s0, acts, args.noise, rng)
        ours += int(A.recovered_exactly(A.abduce(full, obs, acts, args.k), s0))
        orac += int(A.recovered_exactly(A.abduce_true(obs, acts, args.k), s0))
        raw += int(A.recovered_exactly(A.raw_first_frame(obs, args.k), s0))
        deg += int(A.recovered_exactly(A.abduce(degraded, obs, acts, args.k), s0))
        try:
            est, _ = llm.solve(obs, acts, args.noise, args.k)
        except Exception as e:
            print(f"      [llm error: {e}]", flush=True)
            est = None
        if est is not None:
            parsed += 1
            llm_ok += int(A.recovered_exactly(est, s0))

    n = args.n
    print(f"  recover s0 exactly:")
    print(f"    ours (learned MAP)       {ours}/{n}")
    print(f"    oracle (true dynamics)   {orac}/{n}")
    print(f"    LLM ({model})            {llm_ok}/{n}   (parsed {parsed}/{n})")
    print(f"    raw first-frame (floor)  {raw}/{n}")
    print(f"    DEGRADED model (G2)      {deg}/{n}")
    print("\n  -> our reasoner denoises the trajectory with its LEARNED dynamics (MAP over"
          "\n     candidate initial boards); the LLM gets the same noisy trajectory + actions"
          "\n     + rules. Abduction is the inverse mode; report straight whether the LLM can"
          "\n     recover the initial state at all. (Single-frame denoising is the floor.)",
          flush=True)


if __name__ == "__main__":
    run()
