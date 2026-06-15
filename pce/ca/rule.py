"""The size-generalizing representation: a weight-shared LOCAL rule.

This is the structural fix for the audit's "fixed-object codec, zero
generalization" kill. The model is a single radius-1 update law with NO width
parameter: each cell is predicted from its LOCAL neighborhood in relative
coordinates `(prev_i, cur_{i-1}, cur_i, cur_{i+1})`, by ONE shared MLP applied
identically at every cell and every ring width (cortical uniformity / NKS local
rule). Because there is literally no width in the weights, a model fit on small
rings instantiates UNCHANGED on arbitrarily large rings — size-generalization is
structural, not hoped-for.

Training uses per-cell examples pooled across all widths: since the rule is
local and shared, each cell is an i.i.d. example with a 4-dim local feature, and
the model never sees "width" at all.
"""

from __future__ import annotations

import numpy as np


def neighborhood_features(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """(L,) configs -> (L, 4) local features [prev_i, cur_{i-1}, cur_i, cur_{i+1}]."""
    prev = np.asarray(prev, np.float32)
    cur = np.asarray(cur, np.float32)
    left = np.roll(cur, 1)    # cur_{i-1}
    right = np.roll(cur, -1)  # cur_{i+1}
    return np.stack([prev, left, cur, right], axis=-1).astype(np.float32)


class LocalRuleModel:
    """A shared per-cell MLP (4 -> hidden -> 1). Width-agnostic by construction."""

    def __init__(self, hidden: int = 16, lr: float = 1e-2, seed: int = 0):
        import torch
        import torch.nn as nn
        self.torch, self.nn = torch, nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, transitions, steps: int = 1500, batch: int = 512):
        torch = self.torch
        X = np.concatenate([neighborhood_features(p, c) for p, c, n in transitions])
        Y = np.concatenate([np.asarray(n, np.float32) for p, c, n in transitions])
        X = torch.tensor(X)
        Y = torch.tensor(Y).unsqueeze(1)
        n = len(X)
        bce = self.nn.functional.binary_cross_entropy_with_logits
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch, n),))
            loss = bce(self.net(X[idx]), Y[idx])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return self

    def predict_config(self, prev, cur) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            logit = self.net(torch.tensor(neighborhood_features(prev, cur)))
        return (logit.squeeze(-1) > 0).to(torch.int8).numpy()

    def whole_config_accuracy(self, transitions) -> float:
        """Fraction of transitions whose ENTIRE predicted next-config is exact."""
        ok = sum(int(np.array_equal(self.predict_config(p, c), np.asarray(n, np.int8)))
                 for p, c, n in transitions)
        return ok / max(len(transitions), 1)

    def per_cell_accuracy(self, transitions) -> float:
        tot = good = 0
        for p, c, n in transitions:
            pred = self.predict_config(p, c)
            n = np.asarray(n, np.int8)
            good += int((pred == n).sum())
            tot += len(n)
        return good / max(tot, 1)
