"""Head-to-head: a learned-simulation reasoner vs a frontier LLM on forward
execution of an irreducible CA (T1), the core of the reasoning-vs-retrieval test.

Thesis (grounded in the SOTA sweep): on computationally-IRREDUCIBLE iterated
computation, a frontier LLM-as-reasoner degrades with horizon (execution is the
bottleneck — Apple "Illusion of Thinking" arXiv:2506.06941; "Illusion of
Diminishing Returns" arXiv:2509.09677), while a weight-shared LEARNED LOCAL RULE
rolled forward stays exact and generalizes by construction. NOTE (corrected after
scaling): the reducibility dial does NOT bite on FORWARD execution — both rules are
equally laborious step-by-step, so reducibility (a shortcut) doesn't help the LLM
here, and reasoning models (o3) hold to large N. The dial bites on the INVERSE
problem (T3 below): chaos AIDS the inverse, reducibility degenerates it.

Fair-fight rules from the sweep, enforced here:
  * the LLM is given the EXACT rule table and may use a scratchpad (Apple: it still
    collapses) — so a failure is execution, not missing knowledge;
  * NO code execution (no tools passed) — McLeish arXiv:2404.03441: an LLM that
    writes the program wins trivially (that is retrieval of a known algorithm);
  * report per-step accuracy + horizon-to-failure, not just the final answer
    (defuses the context-window artifact, arXiv:2506.09250).

Our reasoner here = M1's learned `LocalRuleModel` rolled forward (it is exact and
size-general by construction); the LLM is called live via pce/llm.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .env import SecondOrderCA
from .rule import neighborhood_features


def rule_table_text(rule: int) -> str:
    """The elementary rule as an explicit 3-neighbourhood truth table."""
    rows = []
    for idx in range(7, -1, -1):
        a, b, c = (idx >> 2) & 1, (idx >> 1) & 1, idx & 1
        rows.append(f"({a},{b},{c})->{(rule >> idx) & 1}")
    return ", ".join(rows)


def forward_prompt(rule: int, prev: np.ndarray, cur: np.ndarray, N: int) -> str:
    L = len(cur)
    return (
        "You are simulating a 1-D second-order cellular automaton on a ring "
        f"(periodic boundary) of width {L}. State is two rows (prev, cur) of bits.\n"
        "Update rule, applied to EVERY cell i (indices wrap mod L):\n"
        f"  f is the elementary rule with table: {rule_table_text(rule)}\n"
        "  next[i] = f(cur[i-1], cur[i], cur[i+1]) XOR prev[i]\n"
        "  then advance: (prev, cur) := (cur, next).\n"
        f"Apply the update {N} times.\n\n"
        f"prev = {''.join(map(str, prev.tolist()))}\n"
        f"cur  = {''.join(map(str, cur.tolist()))}\n\n"
        "Work step by step if needed, but end your answer with a line EXACTLY of the form:\n"
        f"FINAL: <the {L}-bit cur row after {N} steps>\n"
        "Do not use any tools or code; compute it yourself."
    )


def parse_bits(text: str, L: int) -> Optional[np.ndarray]:
    m = re.search(r"FINAL:\s*([01]{%d})" % L, text)
    if not m:
        runs = re.findall(r"[01]{%d}" % L, text)  # fall back to the last L-bit run
        if not runs:
            return None
        return np.array([int(x) for x in runs[-1]], dtype=np.int8)
    return np.array([int(x) for x in m.group(1)], dtype=np.int8)


def true_forward(rule: int, prev, cur, N: int) -> np.ndarray:
    _, c, _ = SecondOrderCA(rule).rollout(prev, cur, N)
    return c


def learned_forward(model, prev, cur, N: int) -> np.ndarray:
    """Our reasoner: roll the LEARNED local rule forward N steps (no true CA)."""
    import torch
    p, c = np.asarray(prev, np.int8), np.asarray(cur, np.int8)
    for _ in range(N):
        feat = neighborhood_features(p, c)
        with torch.no_grad():
            nxt = (model.net(torch.tensor(feat)).squeeze(-1) > 0).to(torch.int8).numpy()
        p, c = c, nxt
    return c


@dataclass
class LLMForwardReasoner:
    client: object   # pce.llm.LLMClient
    max_retries: int = 1

    def solve(self, rule, prev, cur, N: int):
        text = self.client.complete(forward_prompt(rule, prev, cur, N))
        return parse_bits(text, len(cur)), text


def per_step_accuracy(pred_final, true_traj_final) -> float:
    """Bit accuracy of the final config (the whole-config exact-match is stricter)."""
    if pred_final is None:
        return 0.0
    return float((pred_final == true_traj_final).mean())


# --- T3: the INVERSE problem (abduction) — recover the IC from a noisy trajectory ---

def inverse_prompt(rule: int, obs: np.ndarray, noise: float) -> str:
    """obs: (H+1, L) noisy observation of the trajectory [x0, x1, ..., xH]."""
    H1, L = obs.shape
    rows = "\n".join("".join(map(str, obs[t].tolist())) for t in range(H1))
    return (
        f"A 1-D second-order cellular automaton on a ring of {L} cells uses Wolfram rule "
        f"{rule} (table: {rule_table_text(rule)}). The dynamics are "
        f"x[t+1][i] = f(x[t][i-1], x[t][i], x[t][i+1]) XOR x[t-1][i] (indices wrap mod {L}).\n"
        f"Below are {H1} consecutive rows x[0], x[1], ..., x[{H1 - 1}] of a single trajectory, "
        f"but each bit was independently flipped with probability {noise} (observation noise).\n"
        f"{rows}\n\n"
        "The whole trajectory is determined by its first two rows (x[0], x[1]) via the rule. "
        "Using the dynamics to denoise, recover the TRUE first two rows. Reason step by step if "
        f"needed, then end with two lines EXACTLY:\nFINAL_X0: <{L}-bit row>\nFINAL_X1: <{L}-bit row>\n"
        "Do not use any tools or code; reason it out yourself."
    )


def parse_inverse(text: str, L: int):
    import re
    m0 = re.search(r"FINAL_X0:\s*([01]{%d})" % L, text)
    m1 = re.search(r"FINAL_X1:\s*([01]{%d})" % L, text)
    if not (m0 and m1):
        return None
    return (np.array([int(x) for x in m0.group(1)], np.int8),
            np.array([int(x) for x in m1.group(1)], np.int8))


@dataclass
class LLMInverseReasoner:
    client: object

    def solve(self, rule, obs, noise):
        text = self.client.complete(inverse_prompt(rule, obs, noise))
        return parse_inverse(text, obs.shape[1]), text


# --- T2: CONTROL / planning — find <=B flips to make a target pattern appear -------

def control_prompt(rule: int, prev, cur, target, anchor: int, k: int, T: int, B: int) -> str:
    L = len(cur)
    return (
        f"A 1-D second-order cellular automaton on a ring of {L} cells uses Wolfram rule "
        f"{rule} (table: {rule_table_text(rule)}). Dynamics each step: "
        f"next[i] = f(cur[i-1], cur[i], cur[i+1]) XOR prev[i], then (prev,cur) := (cur, next).\n"
        f"Initial state: prev = {''.join(map(str, np.asarray(prev).tolist()))}, "
        f"cur = {''.join(map(str, np.asarray(cur).tolist()))}.\n"
        f"You may INTERVENE: before any step t (0 <= t < {T}), you may flip the value of "
        f"chosen cells of the current 'cur' row. You may flip AT MOST {B} cells in total "
        f"across all steps.\n"
        f"GOAL: after exactly {T} steps, cells {anchor}..{anchor + k - 1} of 'cur' must equal "
        f"the pattern {''.join(map(str, np.asarray(target).tolist()))}.\n"
        f"Find flips (at most {B}) achieving this. Reason step by step, then end with a line "
        f"EXACTLY:\nFLIPS: <space-separated (step,cell) pairs like (0,3) (2,5), or the word NONE>\n"
        "No tools or code; reason it out yourself."
    )


def parse_control_flips(text: str, T: int, L: int, B: int):
    import re
    m = re.search(r"FLIPS:\s*(.+)", text)
    if not m:
        return None
    seg = m.group(1)
    if "NONE" in seg.upper() and "(" not in seg:
        return np.zeros((T, L), dtype=np.int8)
    pairs = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", seg)
    flips = np.zeros((T, L), dtype=np.int8)
    used = 0
    for t, c in pairs:
        t, c = int(t), int(c)
        if 0 <= t < T and 0 <= c < L and used < B:
            flips[t, c] ^= 1
            used += 1
    return flips


@dataclass
class LLMControlReasoner:
    client: object

    def solve(self, rule, prev, cur, target, anchor, k, T, B):
        text = self.client.complete(
            control_prompt(rule, prev, cur, target, anchor, k, T, B))
        return parse_control_flips(text, T, len(cur), B), text
