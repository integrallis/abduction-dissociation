"""Training-seed sweep for the CA 'ours equals oracle' claim (the review's top ask).

The scaling results train the local rule once (model-init seed 0, transition-sampling seed 1) and vary
only the EVALUATION instances, so the reported CIs are instance-level, not training-run variance. This
script closes that gap: it retrains the rule under several (model-init, training-data) seeds and scores
abduction recovery on a SINGLE FIXED evaluation set, against the true-dynamics oracle on that same set.
If ours tracks the oracle across training seeds, 'ours equals oracle' is robust to the training run, not
an artifact of seed 0. CPU-only, deterministic, no API access.

Usage:
    python experiments/ca_seed_sweep.py                 # H=12, noise=0.2, n=60, seeds 0..4
    python experiments/ca_seed_sweep.py --H 18 --n 40 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.ca.env import RULE110, RULE90, sample_transitions  # noqa: E402
from pce.ca.estimate import (TrajectoryDenoiser, enumerate_initials,  # noqa: E402
                             make_noisy_trajectory, true_candidate_trajectories)
from pce.ca.rule import LocalRuleModel  # noqa: E402


def run(argv=None):
    ap = argparse.ArgumentParser(description="CA training-seed sweep: ours-vs-oracle robustness")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.2)
    ap.add_argument("--n", type=int, default=60, help="fixed evaluation instances (shared across seeds)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args(argv)

    print("CA training-seed sweep: is 'ours == oracle' robust to the training run?\n" + "=" * 74)
    print(f"  L={args.L} H={args.H} noise={args.noise}, n={args.n} FIXED eval instances/rule, "
          f"training seeds {args.seeds}")
    print(f"  (each seed varies BOTH model init and the transition-sampling RNG)\n")
    print(f"  {'rule':<22} {'oracle':>8} {'ours per training seed':>28} {'mean±std':>14} {'==oracle?':>10}")

    X0, X1 = enumerate_initials(args.L)
    for rule, name in [(RULE110, "Rule 110 (irreducible)"), (RULE90, "Rule 90 (reducible)")]:
        # one fixed evaluation set, reused across every training seed
        rng = np.random.default_rng(2024)
        insts = []
        for _ in range(args.n):
            x0, traj, obs = make_noisy_trajectory(rule, args.L, args.H, args.noise, rng)
            insts.append((x0, traj[1], np.asarray(obs, np.int8)))
        # true-dynamics oracle (MAP under the true rule) on that fixed set
        true_flat = true_candidate_trajectories(rule, X0, X1, args.H).reshape(2 ** (2 * args.L), -1)
        oracle = 0
        for x0, x1, obs in insts:
            m = (true_flat == obs.reshape(-1)[None]).sum(1)
            e = true_flat[int(np.argmax(m))].reshape(args.H + 1, args.L)
            oracle += int(np.array_equal(e[0], x0) and np.array_equal(e[1], x1))

        rates = []
        for s in args.seeds:
            model = LocalRuleModel(seed=s).fit(
                sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(s + 1)), steps=1500)
            den = TrajectoryDenoiser(model, args.L, args.H)
            ok = 0
            for x0, x1, obs in insts:
                est = den.denoise(obs)
                ok += int(np.array_equal(est[0], x0) and np.array_equal(est[1], x1))
            rates.append(ok)
        arr = np.array(rates) / args.n
        per = " ".join(f"{r}/{args.n}" for r in rates)
        allmatch = "yes" if all(r == oracle for r in rates) else f"max -{oracle - min(rates)}"
        print(f"  {name:<22} {f'{oracle}/{args.n}':>8} {per:>28} "
              f"{f'{arr.mean():.3f}±{arr.std():.3f}':>14} {allmatch:>10}")

    print("\n  Read: if ours equals the oracle at every training seed (or the spread is within the CI),")
    print("  the 'ours == oracle' claim is robust to the training run, not an artifact of seed 0.")


if __name__ == "__main__":
    run()
