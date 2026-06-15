"""S2b — head-to-head: the learned-simulation planner vs a frontier LLM on Sokoban control.

Same fair-fight rules as the CA trilogy (pce/ca/headtohead.py): the LLM is given the EXACT
rules + the board and may use a scratchpad, but NO code execution (planning + simulation are
both its weaknesses — PlanBench/Kambhampati). The board is rendered in standard Sokoban ASCII
(the human-readable form the model has most likely seen). We parse a U/D/L/R plan and verify
it in the TRUE env; our reasoner plans the same instance inside its learned model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .env import AGENT, BOX, GOAL, WALL

# board glyphs and the U/D/L/R -> action map (env DIRS = up,down,left,right = 0,1,2,3)
_MOVE = {"U": 0, "D": 1, "L": 2, "R": 3}


def render_ascii(state: np.ndarray) -> str:
    """Sokoban glyphs (faithful, incl. noise artefacts): # wall, space floor, . goal, $ box,
    * box-on-goal, @ agent, + agent-on-goal, & box+agent (a noisy overlap), % box+agent+goal.
    For valid boards the overlap glyphs never occur."""
    H, W = state.shape[1:]
    rows = []
    for y in range(H):
        row = []
        for x in range(W):
            b, a, g = state[BOX, y, x], state[AGENT, y, x], state[GOAL, y, x]
            if state[WALL, y, x]:
                c = "#"
            elif b and a:
                c = "%" if g else "&"
            elif b:
                c = "*" if g else "$"
            elif a:
                c = "+" if g else "@"
            elif g:
                c = "."
            else:
                c = " "
            row.append(c)
        rows.append("".join(row))
    return "\n".join(rows)


def plan_prompt(state: np.ndarray) -> str:
    n_box = int(state[BOX].sum())
    return (
        "You are solving a Sokoban puzzle on a 2D grid. Symbols:\n"
        "  # = wall, (space) = floor, . = goal, $ = box, * = box on a goal, "
        "@ = you (the agent), + = you on a goal.\n"
        "Each move is one of U (up), D (down), L (left), R (right):\n"
        "  - you step one cell that way if it is floor or a goal;\n"
        "  - if a BOX is in the way and the cell BEYOND it (same direction) is floor/goal, you\n"
        "    PUSH that box one cell and step in; you CANNOT push a box into a wall or another\n"
        "    box, and you cannot pull. Pushes are IRREVERSIBLE (a box in a corner is stuck).\n"
        f"GOAL: end with every one of the {n_box} box(es) on a goal (all goals covered).\n\n"
        "Board (rows top to bottom, columns left to right):\n"
        f"{render_ascii(state)}\n\n"
        "Reason step by step (track where each box and you move), then end with a line EXACTLY:\n"
        "PLAN: <a string of moves from U/D/L/R, e.g. RRULD>\n"
        "Do not use any tools or code; solve it yourself."
    )


def parse_plan(text: str):
    """Extract the U/D/L/R plan -> list of actions (or None)."""
    m = re.search(r"PLAN:\s*([UDLRudlr][UDLRudlr ]*)", text)
    if not m:
        return None
    moves = [c for c in m.group(1).upper() if c in _MOVE]
    return [_MOVE[c] for c in moves] if moves else None


@dataclass
class LLMSokReasoner:
    client: object   # pce.llm.LLMClient

    def solve(self, state: np.ndarray):
        text = self.client.complete(plan_prompt(state))
        return parse_plan(text), text


# --- S3: abduction — recover the true initial board from a NOISY trajectory --------

_RULES = (
    "Sokoban rules: each move U/D/L/R steps the agent one cell that way if it is floor/goal; "
    "if a box is in the way and the cell BEYOND it (same direction) is floor/goal, the box is "
    "PUSHED one cell and the agent steps in; you cannot push into a wall or another box, and "
    "you cannot pull. Walls (#) and goals (.) never move."
)


def abduce_prompt(obs: np.ndarray, actions, noise: float, k_boxes: int) -> str:
    """obs: (T+1,4,H,W) NOISY trajectory. Show each snapshot in ASCII (walls/goals are clean;
    box/agent cells were each independently flipped with prob `noise`) with the action between
    snapshots. Ask for the TRUE initial board (step 0)."""
    H, W = obs.shape[2:]
    names = {0: "U", 1: "D", 2: "L", 3: "R"}
    frames = []
    for t in range(obs.shape[0]):
        head = f"Snapshot {t}:"
        if t < len(actions):
            head = f"Snapshot {t}  (then action {names[int(actions[t])]}):"
        frames.append(head + "\n" + render_ascii(obs[t]))
    glyphs = ("# wall, (space) floor, . goal, $ box, * box-on-goal, @ agent, + agent-on-goal, "
              "& box+agent overlap, % box+agent+goal (the overlaps only appear as NOISE).")
    return (
        f"{_RULES}\n\n"
        f"Below are {obs.shape[0]} NOISY snapshots of a single {H}x{W} game. In every snapshot, "
        f"each box and agent cell was independently flipped with probability {noise} "
        f"(walls # and goals . are shown CORRECTLY). The true boards follow the rules under the "
        f"actions shown. There are exactly {k_boxes} boxes and 1 agent in the true board.\n"
        f"Glyphs: {glyphs}\n\n"
        + "\n\n".join(frames) +
        "\n\nUsing the dynamics to denoise across all snapshots, recover the TRUE INITIAL board "
        "(Snapshot 0). Reason step by step, then end with EXACTLY two lines (0-indexed row,col):\n"
        f"BOXES: {' '.join(['(r,c)'] * k_boxes)}\nAGENT: (r,c)\n"
        "Do not use any tools or code."
    )


def parse_initial(text: str, H: int, W: int, k_boxes: int):
    """Parse BOXES/AGENT coordinate lines -> a (4,H,W) board with only BOX/AGENT set
    (WALL/GOAL filled by the caller). Returns None on malformed output."""
    mb = re.search(r"BOXES:\s*(.+)", text)
    ma = re.search(r"AGENT:\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if not (mb and ma):
        return None
    boxes = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", mb.group(1))
    if len(boxes) < k_boxes:
        return None
    s = np.zeros((4, H, W), np.int8)
    for (r, c) in boxes[:k_boxes]:
        r, c = int(r), int(c)
        if 0 <= r < H and 0 <= c < W:
            s[BOX, r, c] = 1
    ar, ac = int(ma.group(1)), int(ma.group(2))
    if not (0 <= ar < H and 0 <= ac < W):
        return None
    s[AGENT, ar, ac] = 1
    return s


@dataclass
class LLMSokAbducer:
    client: object

    def solve(self, obs, actions, noise, k_boxes):
        text = self.client.complete(abduce_prompt(obs, actions, noise, k_boxes))
        return parse_initial(text, obs.shape[2], obs.shape[3], k_boxes), text
