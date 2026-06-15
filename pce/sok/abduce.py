"""S3 — abduction on Sokoban: recover the initial layout from a NOISY trajectory.

The faithful analog of the CA inverse (pce/ca/estimate.py), now on irreversible 2D dynamics.
A trajectory s_0..s_T is produced by a KNOWN action sequence; we observe each board with the
dynamic planes (BOX, AGENT) independently bit-flipped with probability `noise` (WALL/GOAL are
static structure, observed clean). The whole trajectory is determined by s_0 and the actions,
so a learned-dynamics MAP aggregates evidence across all frames to recover s_0 — where a single
noisy frame cannot.

Inference (no giant state enumeration): the unknowns at s_0 are the placement of the k boxes
and the agent on the known free cells. We ENUMERATE those candidate initial boards, roll each
forward under the known actions with the LEARNED model (batched), and pick the candidate whose
trajectory best matches the noisy observation (a MAP under a symmetric-noise likelihood). The
learned model is the denoising prior; a degraded/wrong model recovers worse than the raw frame.

Irreversibility is the twist vs the (reversible) CA: forward push-dynamics destroy information,
so some wrong initial boards can be observationally close — making the inverse non-trivial.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .env import AGENT, BOX, GOAL, WALL, step


def make_noisy_trajectory(s0: np.ndarray, actions, noise: float, rng):
    """Roll s0 forward under `actions` (true dynamics); flip BOX/AGENT cells with prob `noise`.
    Returns (true_traj, obs) each (T+1,4,H,W); WALL/GOAL planes are left clean."""
    traj = [s0]
    s = s0
    for a in actions:
        s = step(s, int(a))
        traj.append(s)
    traj = np.stack(traj)                                     # (T+1,4,H,W)
    obs = traj.copy()
    for pl in (BOX, AGENT):
        flip = rng.random(obs[:, pl].shape) < noise
        obs[:, pl] = (obs[:, pl] ^ flip).astype(np.int8)
    return traj, obs


def free_cells(template: np.ndarray):
    """Interior non-wall cells (where boxes/agent may sit). WALL/GOAL come from the template."""
    H, W = template.shape[1:]
    return [(y, x) for y in range(H) for x in range(W) if not template[WALL, y, x]]


def enumerate_initials(template: np.ndarray, k_boxes: int, max_candidates: int = 20000):
    """All candidate s_0 boards: k boxes + 1 agent on distinct free cells, with the known
    WALL/GOAL planes from `template`. Returns a (B,4,H,W) int8 stack (capped)."""
    H, W = template.shape[1:]
    cells = free_cells(template)
    base = np.zeros((4, H, W), np.int8)
    base[WALL] = template[WALL]
    base[GOAL] = template[GOAL]
    out = []
    for boxes in combinations(cells, k_boxes):
        bset = set(boxes)
        for (ay, ax) in cells:
            if (ay, ax) in bset:
                continue
            s = base.copy()
            for (by, bx) in boxes:
                s[BOX, by, bx] = 1
            s[AGENT, ay, ax] = 1
            out.append(s)
            if len(out) >= max_candidates:
                return np.stack(out)
    return np.stack(out)


def _rollout_batch(model, states, actions):
    """(B,4,H,W) rolled forward under shared `actions` with the LEARNED model -> (B,T+1,4,H,W)."""
    cur = states
    traj = [cur]
    for a in actions:
        cur = model.predict_next_batch(cur, int(a))
        traj.append(cur)
    return np.stack(traj, axis=1)


def _score(cand_traj, obs):
    """Per-candidate cell-match of the dynamic (BOX,AGENT) planes to the noisy obs, summed over
    all frames (the MAP objective under symmetric noise)."""
    c = cand_traj[:, :, [BOX, AGENT]]                         # (B,T+1,2,H,W)
    o = obs[None, :, [BOX, AGENT]]                            # (1,T+1,2,H,W)
    return (c == o).reshape(c.shape[0], -1).sum(1)


def abduce(model, obs, actions, k_boxes, batched_rollout=_rollout_batch):
    """Recover s_0 by MAP over candidate initial boards rolled forward in the LEARNED model.
    obs: (T+1,4,H,W) noisy. Returns the best-scoring candidate s_0 (4,H,W)."""
    template = obs[0]                                         # WALL/GOAL are clean in obs
    cands = enumerate_initials(template, k_boxes)
    traj = batched_rollout(model, cands, actions)
    scores = _score(traj, obs)
    return cands[int(np.argmax(scores))]


def _true_rollout_batch(_model, states, actions):
    """Oracle rollout with the TRUE dynamics (per-state; for the oracle MAP)."""
    cur = states
    traj = [cur]
    for a in actions:
        cur = np.stack([step(s, int(a)) for s in cur])
        traj.append(cur)
    return np.stack(traj, axis=1)


def abduce_true(obs, actions, k_boxes):
    """Oracle: the same MAP using the TRUE dynamics (upper bound on recoverability)."""
    return abduce(None, obs, actions, k_boxes, batched_rollout=_true_rollout_batch)


def raw_first_frame(obs, k_boxes):
    """Baseline with NO dynamics: clean the noisy first frame to the k highest-evidence box
    cells + the single highest agent cell (best single-frame guess). Shows what the trajectory
    + dynamics buy over just denoising s_0 alone."""
    template = obs[0]
    s = np.zeros_like(template)
    s[WALL] = template[WALL]; s[GOAL] = template[GOAL]
    H, W = template.shape[1:]
    nonwall = ~template[WALL].astype(bool)
    box_ev = (obs[0, BOX] * nonwall).astype(float)
    ag_ev = (obs[0, AGENT] * nonwall).astype(float)
    flat_b = box_ev.reshape(-1)
    for idx in np.argsort(flat_b)[::-1][:k_boxes]:
        if flat_b[idx] > 0:
            s[BOX, idx // W, idx % W] = 1
    ai = int(np.argmax(ag_ev))
    s[AGENT, ai // W, ai % W] = 1
    return s


def recovered_exactly(est, true_s0) -> bool:
    """Exact recovery of the dynamic planes (BOX placement + AGENT position)."""
    return bool(np.array_equal(est[[BOX, AGENT]], true_s0[[BOX, AGENT]]))
