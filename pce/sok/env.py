"""Ground-truth Sokoban world on a 2D grid — the ONLY source of true dynamics.

State is a (4, H, W) int8 stack of binary feature planes:
    plane 0 WALL   (static)   plane 1 BOX    (dynamic)
    plane 2 GOAL   (static)   plane 3 AGENT  (dynamic, exactly one cell)

Dynamics, given an action d in {up,down,left,right} (the SAME local push rule the CA's
weight-shared rule is the 1D analog of):
  - let q = agent + d (target), r = q + d (beyond);
  - if q is out-of-grid or WALL          -> no-op (nothing moves);
  - if q holds a BOX:
        if r is out-of-grid / WALL / BOX  -> no-op (push blocked);
        else                              -> box moves q->r, agent moves p->q;
  - else (q is free floor)               -> agent moves p->q.
Walls and goals never change; a box on a goal is simply BOX and GOAL both set. Pushes are
IRREVERSIBLE and deadlocks are real -> optimal planning is PSPACE-complete (Hearn/Demaine),
which is what keeps reasoning falsifiably required (unlike the Blocksworld trap).

The learner observes only black-box (state, action, next-state) transitions and never reads
this rule. Grids are generated procedurally; the border is always WALL so out-of-grid ==
wall and the local rule needs no special-casing at the boundary (it pads with WALL).
"""

from __future__ import annotations

import numpy as np

WALL, BOX, GOAL, AGENT = 0, 1, 2, 3
# action -> (dy, dx): 0 up, 1 down, 2 left, 3 right
DIRS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int64)
N_ACTIONS = 4


def agent_pos(state: np.ndarray):
    ys, xs = np.where(state[AGENT] == 1)
    return (int(ys[0]), int(xs[0])) if len(ys) else (None, None)


def _free(state: np.ndarray, y: int, x: int) -> bool:
    """A cell the agent or a box can occupy: in-grid, not wall, not box."""
    H, W = state.shape[1:]
    return 0 <= y < H and 0 <= x < W and not state[WALL, y, x] and not state[BOX, y, x]


def step(state: np.ndarray, action: int) -> np.ndarray:
    """One true Sokoban tick. Returns a NEW (4,H,W) state (input untouched)."""
    nxt = state.copy()
    H, W = state.shape[1:]
    ay, ax = agent_pos(state)
    if ay is None:
        return nxt
    dy, dx = DIRS[action]
    qy, qx = ay + dy, ax + dx
    if not (0 <= qy < H and 0 <= qx < W) or state[WALL, qy, qx]:
        return nxt                                  # blocked by wall / edge
    if state[BOX, qy, qx]:                          # a box at the target
        ry, rx = qy + dy, qx + dx
        if not _free(state, ry, rx):
            return nxt                              # push blocked
        nxt[BOX, qy, qx] = 0
        nxt[BOX, ry, rx] = 1
    nxt[AGENT, ay, ax] = 0
    nxt[AGENT, qy, qx] = 1
    return nxt


def solved(state: np.ndarray) -> bool:
    """Every goal covered by a box (and equal counts -> every box on a goal)."""
    box, goal = state[BOX], state[GOAL]
    return int((box & goal).sum()) == int(goal.sum()) == int(box.sum())


def rollout(state: np.ndarray, actions):
    """Apply a sequence of actions, returning the (T+1) list of states."""
    traj = [state]
    s = state
    for a in actions:
        s = step(s, int(a))
        traj.append(s)
    return traj


# --- procedural generation (M1 needs VALID, diverse grids; not necessarily solvable) ----

def random_grid(H: int, W: int, n_boxes: int, rng: np.random.Generator,
                wall_density: float = 0.08) -> np.ndarray:
    """A valid grid: WALL border + sparse interior walls, then agent / boxes / goals placed
    on distinct free floor cells. Solvability is NOT required for learning the dynamics
    (S1); it matters only for the control experiment (S2)."""
    state = np.zeros((4, H, W), np.int8)
    state[WALL, 0, :] = state[WALL, -1, :] = 1
    state[WALL, :, 0] = state[WALL, :, -1] = 1
    interior = [(y, x) for y in range(1, H - 1) for x in range(1, W - 1)]
    for (y, x) in interior:                          # sparse interior walls
        if rng.random() < wall_density:
            state[WALL, y, x] = 1
    floor = [(y, x) for (y, x) in interior if not state[WALL, y, x]]
    rng.shuffle(floor)
    need = 1 + 2 * n_boxes                            # agent + boxes + goals
    if len(floor) < need:
        return random_grid(H, W, n_boxes, rng, wall_density * 0.5)
    pts = floor[:need]
    ay, ax = pts[0]
    state[AGENT, ay, ax] = 1
    for (by, bx) in pts[1:1 + n_boxes]:
        state[BOX, by, bx] = 1
    for (gy, gx) in pts[1 + n_boxes:1 + 2 * n_boxes]:
        state[GOAL, gy, gx] = 1
    return state


def reverse_move(state: np.ndarray, m: int, pull: bool) -> np.ndarray | None:
    """One BACKWARD (pull) move, used only to generate solvable instances. The agent steps
    in direction m to A+m; if `pull` and a box sits at A-m (behind the motion), it is dragged
    into A (the vacated cell). Returns the new state, or None if the agent step is blocked.
    Reversing a pull-sequence yields a valid forward push-solution, so any state reached this
    way from a solved board is guaranteed SOLVABLE."""
    H, W = state.shape[1:]
    ay, ax = agent_pos(state)
    dy, dx = DIRS[m]
    ty, tx = ay + dy, ax + dx
    if not _free(state, ty, tx):                 # agent must be able to step there
        return None
    nxt = state.copy()
    nxt[AGENT, ay, ax] = 0
    nxt[AGENT, ty, tx] = 1
    if pull:
        by, bx = ay - dy, ax - dx                # cell behind the agent's motion
        if 0 <= by < H and 0 <= bx < W and state[BOX, by, bx]:
            nxt[BOX, by, bx] = 0
            nxt[BOX, ay, ax] = 1                 # box dragged into the vacated cell
    return nxt


def generate_solvable_instance(H: int, W: int, n_boxes: int, n_scramble: int,
                               rng: np.random.Generator, wall_density: float = 0.06):
    """Build a SOLVED board (boxes on goals) then scramble it with random backward pull
    moves. Returns (state, scramble_len) where scramble_len is an upper bound on the optimal
    plan length. Guaranteed solvable; non-trivial because a real intervention is required
    (we reject boards already solved after scrambling)."""
    for _attempt in range(200):
        base = np.zeros((4, H, W), np.int8)
        base[WALL, 0, :] = base[WALL, -1, :] = 1
        base[WALL, :, 0] = base[WALL, :, -1] = 1
        interior = [(y, x) for y in range(1, H - 1) for x in range(1, W - 1)]
        for (y, x) in interior:
            if rng.random() < wall_density:
                base[WALL, y, x] = 1
        floor = [(y, x) for (y, x) in interior if not base[WALL, y, x]]
        if len(floor) < n_boxes + 1:
            continue
        rng.shuffle(floor)
        goals = floor[:n_boxes]
        for (gy, gx) in goals:                   # boxes start ON goals (solved)
            base[GOAL, gy, gx] = 1
            base[BOX, gy, gx] = 1
        ay, ax = floor[n_boxes]
        base[AGENT, ay, ax] = 1
        s = base
        moved = 0
        for _ in range(n_scramble):
            m = int(rng.integers(0, N_ACTIONS))
            ns = reverse_move(s, m, pull=bool(rng.integers(0, 2)))
            if ns is not None:
                s = ns
                moved += 1
        if moved >= max(2, n_scramble // 3) and not solved(s):
            return s, moved
    raise RuntimeError("could not generate a solvable, non-trivial instance")


def _is_corner(state, y, x):
    v = state[WALL, y - 1, x] or state[WALL, y + 1, x]
    h = state[WALL, y, x - 1] or state[WALL, y, x + 1]
    return bool(v and h)


def generate_far_instance(H: int, W: int, n_boxes: int, rng: np.random.Generator,
                          min_dist: int = 4, wall_density: float = 0.0, verify=None):
    """A DEPTH-CONTROLLED solvable instance: place boxes at interior, non-corner cells at
    least `min_dist` (Manhattan) from the nearest goal, then keep it only if `verify(state)`
    returns a plan (solvability + a real optimal-length lower bound). This produces genuinely
    deep puzzles (optimal length grows with min_dist) where random shooting fails — the
    regime in which a planner's solve rate is meaningful (the 'never lead with easy
    instances' rule). `verify` is a callable state -> plan|None (the true-model A*); required.

    Returns (state, optimal_len). Avoiding corner placements rules out the trivial
    box-in-corner deadlocks; remaining unsolvable layouts are rejected by `verify`."""
    if verify is None:
        raise ValueError("generate_far_instance needs a verify(state)->plan|None (true A*)")
    for _attempt in range(600):
        s = np.zeros((4, H, W), np.int8)
        s[WALL, 0, :] = s[WALL, -1, :] = 1
        s[WALL, :, 0] = s[WALL, :, -1] = 1
        interior = [(y, x) for y in range(1, H - 1) for x in range(1, W - 1)]
        for (y, x) in interior:
            if wall_density > 0 and rng.random() < wall_density:
                s[WALL, y, x] = 1
        free = [(y, x) for (y, x) in interior if not s[WALL, y, x]]
        if len(free) < 2 * n_boxes + 1:
            continue
        rng.shuffle(free)
        goals = free[:n_boxes]
        cand = [(y, x) for (y, x) in free[n_boxes:]
                if not _is_corner(s, y, x)
                and min(abs(y - gy) + abs(x - gx) for gy, gx in goals) >= min_dist]
        if len(cand) < n_boxes:
            continue
        rng.shuffle(cand)
        boxes = cand[:n_boxes]
        used = set(goals) | set(boxes)
        rest = [c for c in free if c not in used]
        if not rest:
            continue
        for (gy, gx) in goals:
            s[GOAL, gy, gx] = 1
        for (by, bx) in boxes:
            s[BOX, by, bx] = 1
        ay, ax = rest[0]
        s[AGENT, ay, ax] = 1
        if solved(s):
            continue
        plan = verify(s)
        if plan is not None:
            return s, len(plan)
    raise RuntimeError("could not generate a deep solvable instance")


def sample_transitions(grids, per_grid: int, rng: np.random.Generator):
    """Black-box (state, action, next_state) tuples from random valid walks. A walk takes
    `per_grid` steps per grid; every (state, action, next_state) is recorded (including
    no-ops, so the rule learns 'blocked -> copy' too)."""
    out = []
    for g in grids:
        s = g
        for _ in range(per_grid):
            a = int(rng.integers(0, N_ACTIONS))
            ns = step(s, a)
            out.append((s, a, ns))
            s = ns
    return out
