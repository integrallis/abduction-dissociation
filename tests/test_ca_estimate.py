"""Tests: learned dynamics as a denoising prior (a learned component earning its keep).

The decisive assertions: the learned-dynamics MAP estimator denoises a noisy
trajectory far below the raw observation, AND a degraded/wrong model denoises
worse than the raw observation — so the learned dynamics, not a shortcut, carry
the denoising.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from pce.ca.env import RULE90, RULE110, sample_transitions
from pce.ca.estimate import TrajectoryDenoiser, denoise_error
from pce.ca.rule import LocalRuleModel

L, H = 8, 6


def _train(rule, steps=1200, seed=0):
    tr = sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(seed + 1))
    return LocalRuleModel(seed=seed).fit(tr, steps=steps)


def test_learned_dynamics_denoise_below_raw_observation():
    den = TrajectoryDenoiser(_train(RULE110), L, H)
    raw, mp, x0 = denoise_error(den, RULE110, noise=0.10, n=30)
    assert mp < 0.4 * raw          # learned dynamics denoise far below the raw obs
    assert x0 >= 0.6               # recovers the exact initial condition most of the time


def test_wrong_model_denoises_worse_than_raw():
    raw, mp_ok, _ = denoise_error(TrajectoryDenoiser(_train(RULE110), L, H), RULE110, 0.10, n=30)
    _, mp_deg, _ = denoise_error(TrajectoryDenoiser(LocalRuleModel(seed=999), L, H), RULE110, 0.10, n=30)
    _, mp_shuf, _ = denoise_error(TrajectoryDenoiser(_train(RULE90), L, H), RULE110, 0.10, n=30)
    assert mp_ok < raw             # correct model helps
    assert mp_deg > raw            # untrained model hurts (load-bearing)
    assert mp_shuf > raw           # wrong-rule model hurts


def test_denoising_improves_as_noise_drops():
    den = TrajectoryDenoiser(_train(RULE110), L, H)
    _, mp_hi, _ = denoise_error(den, RULE110, noise=0.15, n=30)
    _, mp_lo, _ = denoise_error(den, RULE110, noise=0.05, n=30)
    assert mp_lo < mp_hi
