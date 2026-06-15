"""The size-generalizing representation for Sokoban: a weight-shared LOCAL push-rule.

The 2D analogue of pce/ca/rule.py. ONE MLP predicts a cell's next (BOX, AGENT) from its
local (2R+1)x(2R+1) neighbourhood of (WALL, BOX, AGENT) planes plus the global action
(one-hot, broadcast to every cell). It is applied identically at every cell and every grid
SIZE, with WALL-padding outside the grid — so, exactly as for the CA, there is no grid
dimension anywhere in the weights and a model fit on small grids instantiates UNCHANGED on
arbitrarily large grids. WALL and GOAL are static and copied through.

Radius 2 suffices: a cell's next occupancy depends only on cells up to two steps away along
the action axis (agent at c-2d pushing a box at c-d into c requires c free; agent at c with
c+d a box and c+2d blocked stays put). The 5x5 window covers that for every axis-aligned
action; the MLP learns to ignore the irrelevant corner cells.
"""

from __future__ import annotations

import numpy as np

from .env import AGENT, BOX, GOAL, N_ACTIONS, WALL


def neighborhood_features(state: np.ndarray, action: int, radius: int = 2) -> np.ndarray:
    """(4,H,W) state + action -> (H*W, 3*(2R+1)^2 + N_ACTIONS) per-cell local features.
    Planes used: WALL, BOX, AGENT (GOAL is static, irrelevant to the dynamics). Outside the
    grid is padded with WALL=1 (box/agent=0)."""
    H, W = state.shape[1:]
    R = radius
    planes = state[[WALL, BOX, AGENT]].astype(np.float32)        # (3,H,W)
    padded = np.zeros((3, H + 2 * R, W + 2 * R), np.float32)
    padded[0] = 1.0                                              # wall outside the grid
    padded[:, R:R + H, R:R + W] = planes
    feats = []
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            feats.append(padded[:, R + dy:R + dy + H, R + dx:R + dx + W])  # (3,H,W)
    nb = np.concatenate(feats, axis=0).transpose(1, 2, 0).reshape(H * W, -1)
    act = np.zeros((H * W, N_ACTIONS), np.float32)
    act[:, action] = 1.0
    return np.concatenate([nb, act], axis=1).astype(np.float32)


def _targets(next_state: np.ndarray) -> np.ndarray:
    """(H*W, 2) per-cell next (BOX, AGENT) — the only dynamic planes."""
    H, W = next_state.shape[1:]
    return next_state[[BOX, AGENT]].astype(np.float32).transpose(1, 2, 0).reshape(H * W, 2)


class SokLocalRule:
    """A shared per-cell MLP (3*25+4 -> hidden -> 2). Grid-size-agnostic by construction."""

    def __init__(self, hidden: int = 64, lr: float = 3e-3, radius: int = 2, seed: int = 0):
        import torch
        import torch.nn as nn
        self.torch, self.nn = torch, nn
        self.radius = radius
        torch.manual_seed(seed)
        in_dim = 3 * (2 * radius + 1) ** 2 + N_ACTIONS
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, transitions, steps: int = 4000, batch: int = 1024):
        torch = self.torch
        X = np.concatenate([neighborhood_features(s, a, self.radius) for s, a, n in transitions])
        Y = np.concatenate([_targets(n) for s, a, n in transitions])
        X = torch.tensor(X); Y = torch.tensor(Y)
        n = len(X)
        bce = self.nn.functional.binary_cross_entropy_with_logits
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch, n),))
            loss = bce(self.net(X[idx]), Y[idx])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        return self

    def predict_next(self, state: np.ndarray, action: int) -> np.ndarray:
        """Predict the whole next state by applying the shared rule at every cell. WALL and
        GOAL are copied (static); BOX and AGENT come from the MLP."""
        torch = self.torch
        H, W = state.shape[1:]
        with torch.no_grad():
            logit = self.net(torch.tensor(neighborhood_features(state, action, self.radius)))
        pred = (logit > 0).to(torch.int8).numpy().reshape(H, W, 2)
        nxt = state.copy()
        nxt[BOX] = pred[:, :, 0]
        nxt[AGENT] = pred[:, :, 1]
        return nxt

    def predict_next_batch(self, states: np.ndarray, action: int) -> np.ndarray:
        """Vectorized one-step prediction for a BATCH of states under the SAME action.
        states: (B,4,H,W) -> (B,4,H,W). Used by the abduction MAP (thousands of candidate
        initial boards rolled forward together)."""
        torch = self.torch
        B, _, H, W = states.shape
        R = self.radius
        planes = states[:, [WALL, BOX, AGENT]].astype(np.float32)        # (B,3,H,W)
        padded = np.zeros((B, 3, H + 2 * R, W + 2 * R), np.float32)
        padded[:, 0] = 1.0
        padded[:, :, R:R + H, R:R + W] = planes
        feats = []
        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                feats.append(padded[:, :, R + dy:R + dy + H, R + dx:R + dx + W])
        nb = np.concatenate(feats, axis=1).transpose(0, 2, 3, 1).reshape(B * H * W, -1)
        act = np.zeros((B * H * W, N_ACTIONS), np.float32)
        act[:, action] = 1.0
        X = np.concatenate([nb, act], axis=1).astype(np.float32)
        with torch.no_grad():
            pred = (self.net(torch.tensor(X)) > 0).to(torch.int8).numpy()
        pred = pred.reshape(B, H, W, 2)
        nxt = states.copy()
        nxt[:, BOX] = pred[:, :, :, 0]
        nxt[:, AGENT] = pred[:, :, :, 1]
        return nxt

    def whole_config_accuracy(self, transitions) -> float:
        """Fraction of transitions whose ENTIRE predicted next grid is exact."""
        ok = sum(int(np.array_equal(self.predict_next(s, a), n)) for s, a, n in transitions)
        return ok / max(len(transitions), 1)

    def per_cell_accuracy(self, transitions) -> float:
        good = tot = 0
        for s, a, n in transitions:
            pred = self.predict_next(s, a)
            good += int((pred[[BOX, AGENT]] == n[[BOX, AGENT]]).sum())
            tot += 2 * n.shape[1] * n.shape[2]
        return good / max(tot, 1)
