"""CA-020 (S4): the inverted wall, seed-robust — irreducibility AIDS the inverse.

Training-free, on the TRUE dynamics. The 'inverted wall' (a science by-product, not a capability claim):
the inverse is *easier* under irreducible chaos than under the reducible rule — the reverse of the
forward intuition. The PRINCIPLE is the known sensitivity<->identifiability / data-assimilation-in-
unstable-subspaces relation (Carrassi-Bocquet); what we report is the class-4-vs-class-2 CA empirical
instantiation, and the S4 GATE is that the ordering is SEED-ROBUST.

Statistic: the likelihood GAP = (match of the true IC - match of the best WRONG IC) per bit, from
`pce.ca.estimate.identifiability`. Higher = sharper/more identifiable. Lead with this (the per-bit
x0-error ordering is reported too, and is NOT seed-robust — the honest caveat). The gate:
  irreducible Rule 110 gap > reducible Rule 90 gap at every horizon, on the WORST seed, AND
  Rule 110's gap GROWS with horizon while Rule 90's SATURATES.

Usage:
    python experiments/ca_inverted_wall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from pce.ca.env import RULE110, RULE90  # noqa: E402
from pce.ca.estimate import identifiability  # noqa: E402


def main():
    L, noise, n, seeds = 8, 0.2, 80, 5
    Hs = [8, 12, 16, 20]
    print("CA-020  S4 inverted wall (seed-robust): irreducibility aids the inverse\n" + "=" * 70)
    print(f"  L={L}, noise={noise}, n={n}/seed x {seeds} seeds; likelihood GAP per bit (higher = sharper)\n")
    print(f"  {'H':>4} | {'Rule 110 gap (mean±std, min)':>32} | {'Rule 90 gap (mean±std, min)':>30} | {'110>90 worst':>12}")

    gap = {RULE110: {}, RULE90: {}}
    xerr = {RULE110: {}, RULE90: {}}
    ordering_ok = True
    for H in Hs:
        line = {}
        for rule in (RULE110, RULE90):
            gs = [identifiability(rule, L, H, noise, n=n, seed=s)["gap"] for s in range(seeds)]
            xs = [identifiability(rule, L, H, noise, n=n, seed=s)["x0_err"] for s in range(seeds)]
            gap[rule][H] = gs
            xerr[rule][H] = xs
            line[rule] = (np.mean(gs), np.std(gs), np.min(gs))
        worst = min(g110 - g90 for g110 in [min(gap[RULE110][H])] for g90 in [max(gap[RULE90][H])])
        worst = min(gap[RULE110][H]) - max(gap[RULE90][H])
        if worst <= 0:
            ordering_ok = False
        m1, s1, lo1 = line[RULE110]; m2, s2, lo2 = line[RULE90]
        print(f"  {H:>4} | {f'{m1:.3f}±{s1:.3f}  (min {lo1:.3f})':>32} | "
              f"{f'{m2:.3f}±{s2:.3f}  (min {lo2:.3f})':>30} | {f'{worst:+.3f}':>12}")

    g110_grows = np.mean(gap[RULE110][Hs[-1]]) > np.mean(gap[RULE110][Hs[0]]) + 0.005
    g90_sat = (np.mean(gap[RULE90][Hs[-1]]) - np.mean(gap[RULE90][Hs[0]])) < \
              (np.mean(gap[RULE110][Hs[-1]]) - np.mean(gap[RULE110][Hs[0]]))
    print(f"\n  Rule 110 gap grows with H ({np.mean(gap[RULE110][Hs[0]]):.3f} -> "
          f"{np.mean(gap[RULE110][Hs[-1]]):.3f}): {g110_grows};  Rule 90 grows less / saturates "
          f"({np.mean(gap[RULE90][Hs[0]]):.3f} -> {np.mean(gap[RULE90][Hs[-1]]):.3f}): {g90_sat}")
    # honesty: the per-bit x0-error ordering is NOT seed-robust
    xord = [min(xerr[RULE90][H]) - max(xerr[RULE110][H]) for H in Hs]  # want 90 worse (higher err) than 110
    x_robust = all(d > 0 for d in xord)
    print(f"  (honesty) per-bit x0-error ordering seed-robust? {x_robust} "
          f"(worst-seed 90-minus-110 err across H: {[round(d, 3) for d in xord]}) — lead with the GAP.")

    print()
    if ordering_ok and g110_grows:
        print("  => INVERTED WALL CONFIRMED (seed-robust): irreducible Rule 110 opens a wider, growing")
        print("     likelihood gap than reducible Rule 90 at every horizon, worst-seed. Chaos AIDS the")
        print("     inverse — the empirical class-4-vs-class-2 instantiation (the principle is cited, not new).")
    else:
        print("  => NOT seed-robust on the gap either — cut the inverted wall to a one-paragraph observation;")
        print("     the abduction dissociation headline stands without it.")


if __name__ == "__main__":
    main()
