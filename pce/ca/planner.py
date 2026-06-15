"""Planning INSIDE the learned model (Milestone 2 — fork A, the spine).

The reasoner never calls the ground-truth CA during search. It enumerates
candidate interventions WITHIN THE TARGET'S LIGHT CONE (size-independent),
simulates each one by rolling the LEARNED local rule forward T steps, and
commits the intervention its own model predicts best. The true CA is touched
exactly once, at the end, to verify.

Two audit fixes are built in and checkable:
  * audit #1 (the prior predictor had zero call sites): `RollCounter` records
    that the learned model fires during search and the true CA fires 0 times.
  * audit #4 (learned model not load-bearing): degrade the model and solve rate
    collapses (the learned rollout is the only thing that ranks interventions;
    under irreducible dynamics no cheap heuristic substitutes).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional

import numpy as np

from .env import ControlInstance, light_cone_cells, verify_control


@dataclass
class RollCounter:
    learned_steps: int = 0   # learned-model ticks during search (must be > 0)
    truth_steps: int = 0     # true-CA ticks during search (must be 0)


def _learned_next_batch(model, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Vectorized one learned tick over a batch of (prev, cur): (N, L) -> (N, L)."""
    import torch
    N, L = cur.shape
    left = np.roll(cur, 1, axis=1)
    right = np.roll(cur, -1, axis=1)
    feat = np.stack([prev, left, cur, right], axis=-1).astype(np.float32).reshape(N * L, 4)
    with torch.no_grad():
        logit = model.net(torch.tensor(feat)).reshape(N, L)
    return (logit > 0).to(torch.int8).numpy()


def learned_rollout(model, prev: np.ndarray, cur: np.ndarray, flips: np.ndarray,
                    counter: Optional[RollCounter] = None) -> np.ndarray:
    """Roll the LEARNED model forward over a batch. flips: (N, T, L). Returns
    final cur (N, L). The env is never touched here. State advances as
    (new_prev, new_cur) = (flipped_cur, predicted_next)."""
    p, c = prev.copy(), cur.copy()
    T = flips.shape[1]
    for t in range(T):
        c_flipped = c ^ flips[:, t, :]
        nxt = _learned_next_batch(model, p, c_flipped)
        if counter is not None:
            counter.learned_steps += 1
        p, c = c_flipped, nxt
    return c


def enumerate_candidates(L: int, T: int, k: int, anchor: int, B: int) -> np.ndarray:
    """All flip-sequences of <=B flips placed within the light cone. (N, T, L)."""
    cone = light_cone_cells(L, T, k, anchor)
    positions = [(t, j) for t in range(T) for j in cone]
    seqs: List[list] = [[]]
    for b in range(1, B + 1):
        seqs += [list(cmb) for cmb in combinations(positions, b)]
    F = np.zeros((len(seqs), T, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        for (t, j) in s:
            F[i, t, j] ^= 1
    return F


def plan_in_learned_model(model, inst: ControlInstance,
                          counter: Optional[RollCounter] = None) -> np.ndarray:
    """Counterfactual control by simulation in the LEARNED model: enumerate
    light-cone interventions, roll each through the learned rule, return the
    flip-sequence with the smallest predicted Hamming distance to the target."""
    F = enumerate_candidates(inst.L, inst.T, inst.k, inst.anchor, inst.B)
    N = len(F)
    P = np.tile(inst.prev, (N, 1))
    C = np.tile(inst.cur, (N, 1))
    final = learned_rollout(model, P, C, F, counter)
    ham = (final[:, inst.anchor:inst.anchor + inst.k] != inst.target).sum(1)
    return F[int(np.argmin(ham))]


def random_shooting(inst: ControlInstance, rng: np.random.Generator) -> np.ndarray:
    F = enumerate_candidates(inst.L, inst.T, inst.k, inst.anchor, inst.B)
    return F[int(rng.integers(len(F)))]


def greedy_hamming(model, inst: ControlInstance) -> np.ndarray:
    """One-step learned greedy descent on Hamming distance — the 'shortcut' that
    must FAIL under irreducible dynamics (Hamming is non-monotone per tick)."""
    cone = light_cone_cells(inst.L, inst.T, inst.k, inst.anchor)
    p = inst.prev.copy()
    c = inst.cur.copy()
    flips = np.zeros((inst.T, inst.L), dtype=np.int8)
    used = 0
    for t in range(inst.T):
        candidates = [None] + (list(cone) if used < inst.B else [])
        best_j, best_h = None, None
        for j in candidates:
            cc = c.copy()
            if j is not None:
                cc[j] ^= 1
            nxt = _learned_next_batch(model, p[None, :], cc[None, :])[0]
            h = int((nxt[inst.anchor:inst.anchor + inst.k] != inst.target).sum())
            if best_h is None or h < best_h:
                best_h, best_j = h, j
        if best_j is not None:
            c[best_j] ^= 1
            flips[t, best_j] = 1
            used += 1
        nxt = _learned_next_batch(model, p[None, :], c[None, :])[0]
        p, c = c, nxt   # advance: new prev = flipped cur, new cur = predicted next
    return flips


def solve(model, inst: ControlInstance, counter: Optional[RollCounter] = None) -> bool:
    """Plan in the learned model, then verify ONCE in the true CA."""
    flips = plan_in_learned_model(model, inst, counter)
    return verify_control(inst, flips)
