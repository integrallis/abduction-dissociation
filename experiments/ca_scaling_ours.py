"""CA-017 (S2 scaling, ours side): abduction exact-recovery at scale, multi-seed, with CIs.

S1 confirmed the gate (LLM 0/n is real reasoning; ours reproduces). S2 hardens the OURS side to
defensibility: exact initial-condition recovery (joint x0 AND x1) at n=40/rule x >=5 seeds across a
difficulty axis (observation noise), reported mean +- std with Wilson CIs, alongside the TRUE-DYNAMICS
recoverability ORACLE (the ceiling — ours should sit at it, and the ceiling itself drops as difficulty
rises, so a drop is the task getting harder, not our reasoner failing).

This is the free (CPU) half of S2. The paid half — o3 + a second reasoning model at scale, with the
cached transcripts wired in S1 — establishes the gap with non-overlapping CIs and the 'widens with
difficulty' claim, and is run separately to bound API cost.

Usage:
    python experiments/ca_scaling_ours.py
    python experiments/ca_scaling_ours.py --n 40 --seeds 5 --noises 0.1 0.2 0.3
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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def run(argv=None):
    ap = argparse.ArgumentParser(description="CA-017: S2 ours-side abduction scaling")
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--H", type=int, default=12)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noises", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    args = ap.parse_args(argv)

    print("CA-017  S2 ours-side scaling: exact IC recovery (joint x0 AND x1), multi-seed + CIs\n" + "=" * 76)
    print(f"  L={args.L} H={args.H}, n={args.n}/rule/seed x {args.seeds} seeds; ours = learned-MAP, "
          f"oracle = true-dynamics MAP (the recoverability ceiling)\n")
    print(f"  {'rule':<22} {'noise':>6} {'ours exact (mean±std)':>24} {'oracle ceiling [95% CI]':>26} {'ours==oracle?':>14}")

    X0, X1 = enumerate_initials(args.L)
    for rule, name in [(RULE110, "Rule 110 (irreducible)"), (RULE90, "Rule 90 (reducible)")]:
        model = LocalRuleModel(seed=0).fit(
            sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(1)), steps=1500)
        den = TrajectoryDenoiser(model, args.L, args.H)
        true_flat = true_candidate_trajectories(rule, X0, X1, args.H).reshape(2 ** (2 * args.L), -1)
        for noise in args.noises:
            ours_rates, ok_ours, ok_oracle, tot = [], 0, 0, 0
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed + 1000)
                o = orc = 0
                for _ in range(args.n):
                    x0, traj, obs = make_noisy_trajectory(rule, args.L, args.H, noise, rng)
                    x1 = traj[1]
                    est = den.denoise(obs)
                    o += int(np.array_equal(est[0], x0) and np.array_equal(est[1], x1))
                    m = (true_flat == np.asarray(obs, np.int8).reshape(-1)[None]).sum(1)
                    e = true_flat[int(np.argmax(m))].reshape(args.H + 1, args.L)
                    orc += int(np.array_equal(e[0], x0) and np.array_equal(e[1], x1))
                ours_rates.append(o / args.n); ok_ours += o; ok_oracle += orc; tot += args.n
            lo, hi = wilson(ok_ours, tot)
            olo, ohi = wilson(ok_oracle, tot)
            mean, std = np.mean(ours_rates), np.std(ours_rates)
            match = "yes" if ok_ours == ok_oracle else f"-{ok_oracle - ok_ours}"
            print(f"  {name:<22} {noise:>6.2f} "
                  f"{f'{ok_ours}/{tot}={mean:.3f}±{std:.3f}':>24} "
                  f"{f'{ok_oracle}/{tot} [{olo:.2f},{ohi:.2f}]':>26} {match:>14}")

    print("\n  Read: ours should TRACK the oracle (recoverability ceiling) at every noise — the ceiling")
    print("  itself falls as noise rises (the task gets genuinely harder/ambiguous), so 'ours < 1.0' at")
    print("  high noise is the task, not the reasoner, exactly when ours == oracle. The LLM gap and")
    print("  'widens with difficulty' are the paid o3-at-scale half (cached transcripts wired in S1).")


if __name__ == "__main__":
    run()
