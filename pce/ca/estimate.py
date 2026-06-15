"""Learned dynamics as a denoising prior — where a learned model earns its keep.

Everywhere else in this foundation the learned model could be matched by a simpler
baseline (the obs-space rule generalizes; spatial voting denoises). Here it
cannot: given a NOISY TRAJECTORY of the autonomous CA (one observation per
timestep, NO spatial redundancy), the only way to denoise is to exploit the
cross-time dynamical constraint — the whole trajectory is determined by (x0, x1)
and the rule. A learned-dynamics MAP estimator aggregates evidence across all
timesteps to recover the true state; voting/spatial methods cannot (each timestep
is a different state, so there is nothing to vote over).

Chaos HELPS here: under the irreducible Rule 110, a wrong (x0, x1) diverges within
a few steps, so the likelihood is sharp and the true initial condition is
identifiable from noisy data. And the learned model is provably load-bearing — a
degraded or wrong-rule model denoises WORSE than the raw observation.

Scope: the MAP estimate enumerates initial conditions (2^(2L)), feasible for small
L (<=10). Scaling to larger L needs amortized inference (a learned estimator
network) rather than exhaustive enumeration. That is the next step, flagged honestly.
"""

from __future__ import annotations

import numpy as np

from .env import SecondOrderCA
from .planner import _learned_next_batch


def enumerate_initials(L: int):
    """All (x0, x1) pairs in {0,1}^L x {0,1}^L. Feasible only for small L."""
    single = ((np.arange(2 ** L)[:, None] >> np.arange(L)) & 1).astype(np.int8)
    X0 = np.repeat(single, 2 ** L, axis=0)
    X1 = np.tile(single, (2 ** L, 1))
    return X0, X1


def candidate_trajectories(model, X0, X1, H):
    """Roll every (x0, x1) forward H steps with the LEARNED rule -> (N, H+1, L)."""
    traj = [X0.copy(), X1.copy()]
    p, c = X0, X1
    for _ in range(H - 1):
        nxt = _learned_next_batch(model, p, c)
        traj.append(nxt)
        p, c = c, nxt
    return np.stack(traj, axis=1)


def true_trajectory(rule: int, x0, x1, H):
    ca = SecondOrderCA(rule)
    xs = [np.asarray(x0, np.int8), np.asarray(x1, np.int8)]
    p, c = xs[0], xs[1]
    for _ in range(H - 1):
        n = ca.next_config(p, c)
        xs.append(n)
        p, c = c, n
    return np.stack(xs)


def make_noisy_trajectory(rule: int, L: int, H: int, noise: float, rng):
    x0, x1 = rng.integers(0, 2, size=(2, L)).astype(np.int8)
    traj = true_trajectory(rule, x0, x1, H)
    obs = (traj ^ (rng.random(traj.shape) < noise)).astype(np.int8)
    return x0, traj, obs


class TrajectoryDenoiser:
    """MAP trajectory denoiser using the LEARNED dynamics as the likelihood.
    Precomputes candidate rollouts once (per model, H) and re-scores per obs."""

    def __init__(self, model, L: int, H: int):
        self.L, self.H = L, H
        X0, X1 = enumerate_initials(L)
        self.cand = candidate_trajectories(model, X0, X1, H)   # (N, H+1, L)

    def denoise(self, obs) -> np.ndarray:
        """Return the learned-rollout trajectory best matching the noisy obs."""
        flat = self.cand.reshape(self.cand.shape[0], -1)
        match = (flat == np.asarray(obs, np.int8).reshape(-1)[None]).sum(1)
        return self.cand[int(np.argmax(match))]


def true_candidate_trajectories(rule: int, X0, X1, H):
    """Roll every (x0,x1) forward H steps with the TRUE rule -> (N, H+1, L)."""
    ca = SecondOrderCA(rule)
    traj = [X0.copy(), X1.copy()]
    p, c = X0, X1
    for _ in range(H - 1):
        nxt = ca.next_config(p, c)
        traj.append(nxt)
        p, c = c, nxt
    return np.stack(traj, axis=1)


def identifiability(rule: int, L: int, H: int, noise: float, n: int = 120, seed: int = 0):
    """Intrinsic difficulty of the inverse problem (no learned model, no encoder):
    enumerate all ICs, measure for n noisy trajectories the MAP x0-recovery error,
    the mean LIKELIHOOD GAP = (match of true IC - match of best WRONG IC) per bit
    (higher = sharper/more identifiable), and the mean number of ICs TIED at the
    max score (>1 => multimodal/ambiguous posterior). Feasible only for small L."""
    W = 1 << np.arange(L)
    X0, X1 = enumerate_initials(L)
    cand = true_candidate_trajectories(rule, X0, X1, H).reshape(2 ** (2 * L), -1)
    rng = np.random.default_rng(seed)
    x0err = cells = 0
    gaps, ties = [], []
    for _ in range(n):
        a, b = rng.integers(0, 2, size=(2, L)).astype(np.int8)
        ti = int(a.dot(W)) * (2 ** L) + int(b.dot(W))
        obs = (cand[ti] ^ (rng.random(cand.shape[1]) < noise)).astype(np.int8)
        match = (cand == obs[None]).sum(1)
        best = int(np.argmax(match))
        x0err += int((X0[best] != a).sum())
        cells += L
        m2 = match.copy(); m2[ti] = -1
        gaps.append((int(match[ti]) - int(m2.max())) / cand.shape[1])
        ties.append(int((match == match.max()).sum()))
    return {"x0_err": x0err / cells, "gap": float(np.mean(gaps)), "ties": float(np.mean(ties))}


def true_posterior_marginals(rule: int, obs, H: int, noise: float):
    """Exact per-cell posterior P(x0[i]=1 | obs), P(x1[i]=1 | obs) by enumerating
    all initial conditions (feasible for small L). The gold standard a calibrated
    amortized estimator should match; where this is ~0.5 the IC is genuinely
    ambiguous (e.g. Rule 90's coset degeneracy)."""
    L = len(obs[0]) if hasattr(obs[0], "__len__") else int(round(len(np.asarray(obs).reshape(-1)) / (H + 1)))
    obs = np.asarray(obs, np.int8).reshape(-1)
    X0, X1 = enumerate_initials(L)
    cand = true_candidate_trajectories(rule, X0, X1, H).reshape(2 ** (2 * L), -1)
    beta = np.log((1 - noise) / noise)
    lp = beta * (cand == obs[None]).sum(1).astype(np.float64)
    lp -= lp.max()
    p = np.exp(lp); p /= p.sum()
    return p @ X0, p @ X1


def difference_growth(rule: int, L: int, H: int, max_hamming: int = 2,
                      n_base: int = 40, seed: int = 0) -> float:
    """The CAUSAL mechanism behind the identifiability asymmetry: the minimum (over
    nearby initial conditions, i.e. small-Hamming perturbations delta) per-bit
    Hamming distance between an IC's trajectory and its perturbed twin's, averaged
    over base ICs. This is the divergence of the HARDEST-to-distinguish rival IC,
    which sets the identifiability-gap floor.

    Chaotic (irreducible) rules amplify every perturbation -> this grows with H
    (no persistent low-growth mode) -> the inverse keeps sharpening. Linear/additive
    (reducible) rules have a kernel/low-growth difference-mode whose trajectory stays
    bounded for all H -> a fixed gap floor -> a saturating, ambiguous inverse. This
    is the Spoto-Milani (arXiv:1506.03221) exponential-vs-polynomial law and the
    data-assimilation unstable-subspace picture (Carrassi/Bocquet arXiv:2010.07063),
    instantiated on our second-order CA (whose full map is bijective, so the
    mechanism is difference-trajectory growth, not single-step preimage count)."""
    from itertools import combinations
    ca = SecondOrderCA(rule)
    rng = np.random.default_rng(seed)
    deltas = []
    for h in range(1, max_hamming + 1):
        for combo in combinations(range(2 * L), h):
            d0 = np.zeros(L, np.int8); d1 = np.zeros(L, np.int8)
            for idx in combo:
                (d0 if idx < L else d1)[idx % L] ^= 1
            deltas.append((d0, d1))
    bases = [rng.integers(0, 2, size=(2, L)).astype(np.int8) for _ in range(n_base)]

    def traj(x0, x1):
        xs = [x0, x1]; p, c = x0, x1
        for _ in range(H - 1):
            nxt = ca.next_config(p, c); xs.append(nxt); p, c = c, nxt
        return np.stack(xs)

    best = np.inf
    for d0, d1 in deltas:
        tot = sum((traj(a, b) != traj(a ^ d0, b ^ d1)).sum() for a, b in bases)
        best = min(best, tot / n_base)
    return best / ((H + 1) * L)


def denoise_error(denoiser: TrajectoryDenoiser, rule: int, noise: float,
                  n: int = 40, seed: int = 0):
    """Return (raw_obs_error, map_error, exact_x0_recovery_rate)."""
    rng = np.random.default_rng(seed)
    raw = mp = x0_ok = cells = 0
    for _ in range(n):
        x0, traj, obs = make_noisy_trajectory(rule, denoiser.L, denoiser.H, noise, rng)
        est = denoiser.denoise(obs)
        raw += int((obs != traj).sum())
        mp += int((est != traj).sum())
        cells += traj.size
        x0_ok += int(np.array_equal(est[0], x0))
    return raw / cells, mp / cells, x0_ok / n
