"""Sokoban CONTROL: the classical A* (true-model) baseline, made explicit.

The control table foregrounds ours / random / LLM, but a reviewer fairly asks for
the classical reference in the table rather than only in code comments. A* over the
TRUE model solves these instances by construction (build_bins admits an instance
only if plan_in_true_model finds a plan, which also yields its optimal length), so
this is NOT a "beats A*" claim -- A* is the upper reference the matched comparison
sits under. This script computes it explicitly on the SAME seeded 18 instances the
head-to-head uses (by importing build_bins) and reports the solve count.

Usage:
    python experiments/sok_control_astar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.sok import planner as P  # noqa: E402
from experiments.sok_control_llm import BINS, build_bins  # noqa: E402


def run():
    rng = np.random.default_rng(100)          # identical seed to sok_control_llm.run()
    bins = build_bins(rng, 6)
    print("  Sokoban CONTROL -- classical A* (true-model) reference on the SAME 18 instances:")
    print(f"  {'difficulty':<16} {'n':>3} {'A* solved':>10} {'opt len (min..max)':>20}")
    tot_ok = tot_n = 0
    for b in BINS:
        insts = bins.get(b, [])
        ok = 0
        ols = []
        for s, ol in insts:
            pl, _, _ = P.plan_in_true_model(s, max_expansions=20000)
            ok += int(pl is not None and P.verify_plan(s, pl))
            ols.append(int(ol))
        tot_ok += ok; tot_n += len(insts)
        rng_str = f"{min(ols)}..{max(ols)}" if ols else "-"
        print(f"  {b:<16} {len(insts):>3} {f'{ok}/{len(insts)}':>10} {rng_str:>20}")
    print(f"  {'TOTAL':<16} {tot_n:>3} {f'{tot_ok}/{tot_n}':>10}")
    print(f"\n  A* (true model) solves {tot_ok}/{tot_n}: the classical upper reference the matched")
    print("  comparison sits under. ours == A* here; the contribution is the abduction axis, not control.")


if __name__ == "__main__":
    run()
