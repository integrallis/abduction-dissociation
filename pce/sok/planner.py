"""S2 — plan INSIDE the learned Sokoban model (the M2 spine ported to 2D).

The planner searches for an action sequence that solves the puzzle by rolling a transition
MODEL forward; the true environment is touched only ONCE at the end to verify the committed
plan. The model is pluggable, so the SAME search drives:
  - the LEARNED-model planner (transitions from SokLocalRule.predict_next),
  - the true-model ORACLE (an upper bound),
  - the classical A*+Manhattan REFERENCE (the anti-Blocksworld-trap gate G3).

Audit-#1 discipline (ported): a RollCounter records learned-model vs true-env ticks; during
the learned search true-env ticks MUST be 0. Audit-#4/G2: degrading the learned model must
collapse the solve rate toward random — proving the learned model is load-bearing.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .env import AGENT, BOX, GOAL, N_ACTIONS, WALL, agent_pos, solved, step


@dataclass
class RollCounter:
    learned: int = 0
    true: int = 0


def _key(state: np.ndarray):
    ay, ax = agent_pos(state)
    boxes = frozenset(map(tuple, np.argwhere(state[BOX] == 1)))
    return (ay, ax, boxes)


def _goal_coords(state):
    return np.argwhere(state[GOAL] == 1)


def manhattan_h(state: np.ndarray, goals: np.ndarray) -> int:
    """Sum over boxes of the distance to the nearest goal (a cheap, standard heuristic)."""
    boxes = np.argwhere(state[BOX] == 1)
    if len(boxes) == 0 or len(goals) == 0:
        return 0
    d = np.abs(boxes[:, None, :] - goals[None, :, :]).sum(-1)   # (nbox, ngoal)
    return int(d.min(axis=1).sum())


def best_first_search(start: np.ndarray, step_fn, max_expansions: int = 4000,
                      heuristic: bool = True):
    """A*/greedy search to a solved state using step_fn(state, action) -> next_state.
    Returns (plan, expansions) or (None, expansions). step_fn is where the model is touched
    (the caller counts the ticks)."""
    goals = _goal_coords(start)
    h0 = manhattan_h(start, goals) if heuristic else 0
    pq = [(h0, 0, 0, start, [])]                 # (f, g, tiebreak, state, plan)
    seen = {_key(start): 0}
    tie = 1
    exp = 0
    while pq and exp < max_expansions:
        f, g, _, state, plan = heapq.heappop(pq)
        if solved(state):
            return plan, exp
        exp += 1
        for a in range(N_ACTIONS):
            nxt = step_fn(state, a)
            if nxt is None:
                continue
            k = _key(nxt)
            ng = g + 1
            if k in seen and seen[k] <= ng:
                continue
            seen[k] = ng
            h = manhattan_h(nxt, goals) if heuristic else 0
            heapq.heappush(pq, (ng + h, ng, tie, nxt, plan + [a]))
            tie += 1
    return None, exp


def plan_in_learned_model(model, start: np.ndarray, max_expansions: int = 4000,
                          counter: RollCounter | None = None):
    """Plan by rolling the LEARNED rule forward. The true env is NOT touched here."""
    counter = counter or RollCounter()

    def step_fn(state, action):
        counter.learned += 1
        return model.predict_next(state, action)

    plan, exp = best_first_search(start, step_fn, max_expansions)
    return plan, counter, exp


def plan_in_true_model(start: np.ndarray, max_expansions: int = 4000,
                       heuristic: bool = True, counter: RollCounter | None = None):
    """Oracle / A*-reference: plan using the TRUE dynamics (heuristic=True => A*+Manhattan)."""
    counter = counter or RollCounter()

    def step_fn(state, action):
        counter.true += 1
        return step(state, action)

    plan, exp = best_first_search(start, step_fn, max_expansions, heuristic=heuristic)
    return plan, counter, exp


def random_shooting(start: np.ndarray, n_tries: int, max_len: int, rng) -> list | None:
    """Floor baseline: random action sequences, replayed in the true env."""
    for _ in range(n_tries):
        s = start
        plan = []
        for _ in range(max_len):
            a = int(rng.integers(0, N_ACTIONS))
            s = step(s, a)
            plan.append(a)
            if solved(s):
                return plan
    return None


def verify_plan(start: np.ndarray, plan) -> bool:
    """Replay a plan in the TRUE env (the only place ground truth is touched in a solve)."""
    if plan is None:
        return False
    s = start
    for a in plan:
        s = step(s, int(a))
    return solved(s)
