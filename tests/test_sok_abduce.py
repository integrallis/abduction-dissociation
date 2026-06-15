"""Sokoban S3 (abduction) — recover s0 from a noisy trajectory. Deterministic invariants."""

import numpy as np
import pytest

from pce.sok import abduce as A
from pce.sok import env as E
from pce.sok.headtohead import abduce_prompt, parse_initial
from pce.sok.rule import SokLocalRule


@pytest.fixture(scope="module")
def rule():
    rng = np.random.default_rng(0)
    grids = [E.random_grid(int(rng.integers(6, 9)), int(rng.integers(6, 9)),
                           int(rng.integers(2, 4)), rng) for _ in range(250)]
    return SokLocalRule(hidden=64, seed=0).fit(E.sample_transitions(grids, 12, rng), steps=3000)


def test_noisy_trajectory_keeps_walls_goals_clean():
    rng = np.random.default_rng(1)
    s0 = E.random_grid(6, 6, 2, rng)
    acts = [int(rng.integers(0, 4)) for _ in range(5)]
    true, obs = A.make_noisy_trajectory(s0, acts, 0.3, rng)
    assert obs.shape == true.shape and obs.shape[0] == 6
    assert np.array_equal(obs[:, E.WALL], true[:, E.WALL])     # walls untouched by noise
    assert np.array_equal(obs[:, E.GOAL], true[:, E.GOAL])     # goals untouched
    assert not np.array_equal(obs[:, E.BOX], true[:, E.BOX])   # boxes were corrupted


def test_parse_initial_round_trip():
    rng = np.random.default_rng(2)
    s0 = E.random_grid(6, 6, 2, rng)
    bxs = np.argwhere(s0[E.BOX] == 1)
    ag = np.argwhere(s0[E.AGENT] == 1)[0]
    ans = "BOXES: " + " ".join(f"({r},{c})" for r, c in bxs) + f"\nAGENT: ({ag[0]},{ag[1]})"
    assert A.recovered_exactly(parse_initial(ans, 6, 6, 2), s0)
    assert parse_initial("no coords here", 6, 6, 2) is None


def test_prompt_shows_trajectory_and_rules():
    rng = np.random.default_rng(3)
    s0 = E.random_grid(6, 6, 2, rng)
    acts = [int(rng.integers(0, 4)) for _ in range(4)]
    _, obs = A.make_noisy_trajectory(s0, acts, 0.12, rng)
    prompt = abduce_prompt(obs, acts, 0.12, 2)
    assert "Snapshot 0" in prompt and "BOXES:" in prompt and "noisy" in prompt.lower()


def test_learned_map_recovers_better_than_raw_frame(rule):
    """The learned-dynamics MAP recovers s0 from the noisy trajectory; the single-frame
    baseline (no dynamics) does not — the trajectory + dynamics are what denoise."""
    r = np.random.default_rng(7)
    ours = raw = 0
    for _ in range(6):
        s0 = E.random_grid(6, 6, 2, r)
        acts = [int(r.integers(0, 4)) for _ in range(6)]
        _, obs = A.make_noisy_trajectory(s0, acts, 0.1, r)
        ours += int(A.recovered_exactly(A.abduce(rule, obs, acts, 2), s0))
        raw += int(A.recovered_exactly(A.raw_first_frame(obs, 2), s0))
    assert ours >= 4          # recovers most
    assert ours > raw         # and beats the single-frame floor
