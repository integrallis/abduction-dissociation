"""Ground-truth second-order reversible cellular automaton on a ring.

State is a pair (prev, cur) of binary configurations in {0,1}^L with periodic
boundary. The dynamics are a second-order lift of an elementary radius-1 rule f:

    next_i = f(cur_{i-1}, cur_i, cur_{i+1})  XOR  prev_i
    new state = (cur, next)

The XOR-with-prev makes the map REVERSIBLE (information-preserving: given
(cur, next) you recover prev = next XOR f(cur)), which keeps multi-step rollouts
well-posed, while the underlying class-4 rule (default Rule 110) keeps it
computationally IRREDUCIBLE. The SAME rule is applied at every cell and every L
(cortical uniformity / NKS local rule) — the dynamics are width-independent by
construction, which is the whole point.

This module is the ONLY source of true dynamics. The learner observes only
black-box (state, next-state) transitions from it and never reads the rule code.
"""

from __future__ import annotations

import numpy as np

RULE110 = 110   # Wolfram class 4, Turing-complete -> irreducible
RULE90 = 90     # additive (f(a,b,c) = a XOR c) -> reducible control


class SecondOrderCA:
    def __init__(self, rule: int = RULE110):
        self.rule = int(rule)

    def f(self, cur: np.ndarray) -> np.ndarray:
        """Elementary radius-1 rule applied to a configuration (circular)."""
        cur = np.asarray(cur, dtype=np.int64)
        left = np.roll(cur, 1, axis=-1)    # left[i] = cur[i-1]
        right = np.roll(cur, -1, axis=-1)  # right[i] = cur[i+1]
        idx = (left << 2) | (cur << 1) | right
        return ((self.rule >> idx) & 1).astype(np.int8)

    def next_config(self, prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
        """The next configuration (the prediction target for milestone 1)."""
        return (self.f(cur) ^ np.asarray(prev, dtype=np.int8)).astype(np.int8)

    def step(self, prev: np.ndarray, cur: np.ndarray, flip: np.ndarray | None = None):
        """One controlled tick: apply `flip` to cur, then advance. Returns
        (new_prev, new_cur) = (flipped_cur, next)."""
        c = cur if flip is None else (np.asarray(cur, np.int8) ^ np.asarray(flip, np.int8))
        nxt = self.next_config(prev, c)
        return np.asarray(c, np.int8), nxt

    def reverse_step(self, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        """Recover prev from (cur, next) — the map is reversible."""
        return (np.asarray(nxt, np.int8) ^ self.f(cur)).astype(np.int8)

    def rollout(self, prev, cur, steps: int, flips: np.ndarray | None = None):
        """Run `steps` controlled ticks. flips: optional (steps, L) array."""
        p, c = np.asarray(prev, np.int8), np.asarray(cur, np.int8)
        traj = [(p, c)]
        for t in range(steps):
            fl = None if flips is None else flips[t]
            p, c = self.step(p, c, fl)
            traj.append((p, c))
        return p, c, traj


def random_configs(L: int, n: int, rng: np.random.Generator):
    return (rng.integers(0, 2, size=(n, L), dtype=np.int8),
            rng.integers(0, 2, size=(n, L), dtype=np.int8))


def sample_transitions(rule: int, widths, per_width: int, rng: np.random.Generator):
    """Black-box (prev, cur, next) full-configuration transitions across widths.

    The learner sees only these tuples — never the rule code, never the width as
    a feature. Each element is (prev, cur, next), each a length-L int8 array.
    """
    ca = SecondOrderCA(rule)
    out = []
    for L in widths:
        prev, cur = random_configs(L, per_width, rng)
        nxt = ca.next_config(prev, cur)
        out.extend((prev[i], cur[i], nxt[i]) for i in range(per_width))
    return out


# --- goal-directed control instances (milestone 2) ------------------------------

from dataclasses import dataclass


def light_cone_cells(L: int, T: int, k: int, anchor: int):
    """Cells that can influence the target window [anchor, anchor+k) within T
    ticks (radius-1 rule => +/-1 per step). Width k+2T, INDEPENDENT of ring size,
    so intervention search stays size-independent — local reasoning that
    generalizes across width."""
    return sorted({(anchor + d) % L for d in range(-T, k + T)})


@dataclass
class ControlInstance:
    rule: int
    prev: np.ndarray
    cur: np.ndarray
    target: np.ndarray   # the k-cell pattern required at the anchor window at step T
    anchor: int
    k: int
    T: int
    B: int               # flip budget

    @property
    def L(self) -> int:
        return len(self.cur)


def generate_control_instance(rule: int, L: int, T: int, B: int, k: int,
                              anchor: int, rng: np.random.Generator) -> ControlInstance:
    """Forward-generate a SOLVABLE, NONTRIVIAL control instance: apply a random
    witnessed flip-sequence (<=B flips within the light cone) to define the
    target; reject instances the autonomous (no-intervention) rollout already
    satisfies (so a real intervention is required)."""
    ca = SecondOrderCA(rule)
    cone = light_cone_cells(L, T, k, anchor)
    for _ in range(2000):
        prev, cur = random_configs(L, 1, rng)
        prev, cur = prev[0], cur[0]
        nflip = int(rng.integers(1, B + 1))
        fl = np.zeros((T, L), dtype=np.int8)
        for t, j in zip(rng.integers(0, T, size=nflip), rng.choice(cone, size=nflip)):
            fl[int(t), int(j)] ^= 1
        _, c_t, _ = ca.rollout(prev, cur, T, fl)
        target = c_t[anchor:anchor + k].copy()
        _, c_auto, _ = ca.rollout(prev, cur, T)
        if not np.array_equal(c_auto[anchor:anchor + k], target):
            return ControlInstance(rule, prev, cur, target, anchor, k, T, B)
    raise RuntimeError("could not generate a nontrivial control instance")


def verify_control(inst: ControlInstance, flips: np.ndarray) -> bool:
    """Replay a flip-sequence in the TRUE CA and check the target pattern. The
    only place the ground-truth dynamics are touched during a solve."""
    ca = SecondOrderCA(inst.rule)
    _, c_t, _ = ca.rollout(inst.prev, inst.cur, inst.T, flips)
    return bool(np.array_equal(c_t[inst.anchor:inst.anchor + inst.k], inst.target))
