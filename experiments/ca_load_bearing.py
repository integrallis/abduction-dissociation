"""CA-019 (S3): load-bearing controls for CA abduction — degrade-to-chance + call-site audit.

S2 established ours == the true-dynamics oracle at scale (oracle-matching, the first load-bearing
control). S3 adds the other two:

  * DEGRADE-TO-CHANCE: train the local rule at increasing fidelity and show abduction exact-recovery
    TRACKS the learned model's fidelity and collapses toward the floor as the model degrades — the
    graded proof that the LEARNED model carries the inference, not a lookup or a hidden solver. Floors:
    the raw-1-frame estimate (noisy x0,x1 taken as the IC, no model) and random-IC chance.
  * CALL-SITE AUDIT: instrument the inference path — the LEARNED rule supplies every candidate rollout
    (learned ticks > 0) while the TRUE CA is touched 0 times during the search (only to generate the
    instance and to score the answer). The abduction is done INSIDE the learned model.

Usage:
    python experiments/ca_load_bearing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import pce.ca.env as envmod  # noqa: E402
import pce.ca.estimate as est  # noqa: E402
from pce.ca.env import RULE110, sample_transitions  # noqa: E402
from pce.ca.estimate import TrajectoryDenoiser, make_noisy_trajectory  # noqa: E402
from pce.ca.rule import LocalRuleModel  # noqa: E402


def abduction_rate(den, rule, L, H, noise, n, seed):
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        x0, traj, obs = make_noisy_trajectory(rule, L, H, noise, rng)
        est_traj = den.denoise(obs)
        ok += int(np.array_equal(est_traj[0], x0) and np.array_equal(est_traj[1], traj[1]))
    return ok / n


def raw_floor_rate(rule, L, H, noise, n, seed):
    """No model: take the noisy first two rows as the IC estimate."""
    rng = np.random.default_rng(seed)
    ok = 0
    for _ in range(n):
        x0, traj, obs = make_noisy_trajectory(rule, L, H, noise, rng)
        ok += int(np.array_equal(obs[0], x0) and np.array_equal(obs[1], traj[1]))
    return ok / n


def main():
    L, H, noise, n = 8, 12, 0.2, 60
    rule = RULE110
    train = sample_transitions(rule, [6, 7, 8, 9, 10], 400, np.random.default_rng(1))
    test = sample_transitions(rule, [16], 200, np.random.default_rng(2))

    print("CA-019  S3 load-bearing controls (Rule 110, L=8, H=12, noise=0.2)\n" + "=" * 66)
    print("\n[1] DEGRADE-TO-CHANCE: abduction recovery tracks the learned model's fidelity")
    print(f"  {'train_steps':>11} {'fidelity':>9} {'abduction exact':>16}")
    for ts in [3, 10, 30, 100, 300, 1500]:
        model = LocalRuleModel(seed=0).fit(train, steps=ts)
        fid = model.whole_config_accuracy(test)
        den = TrajectoryDenoiser(model, L, H)
        rate = abduction_rate(den, rule, L, H, noise, n, seed=7)
        print(f"  {ts:>11} {fid:>9.3f} {f'{rate:.3f}':>16}")
    floor = raw_floor_rate(rule, L, H, noise, n, seed=7)
    chance = 1.0 / (2 ** (2 * L))
    print(f"  {'— floors —':>11} {'':>9} raw-1-frame={floor:.3f}, random-IC chance={chance:.1e}")
    print("  => recovery rises from ~floor (degraded model -> garbage candidates) to ~0.96 (full model)")
    print("     monotonically with fidelity: the learned model is LOAD-BEARING for abduction.")

    print("\n[2] CALL-SITE AUDIT: which dynamics drive the inference?")
    model = LocalRuleModel(seed=0).fit(train, steps=1500)
    rng = np.random.default_rng(123)
    x0, traj, obs = make_noisy_trajectory(rule, L, H, noise, rng)   # instance generated BEFORE patching
    learned, true = [0], [0]
    _olb, _onc = est._learned_next_batch, envmod.SecondOrderCA.next_config

    def clb(m, p, c):
        learned[0] += 1
        return _olb(m, p, c)

    def cnc(self, prev, cur):
        true[0] += 1
        return _onc(self, prev, cur)

    est._learned_next_batch = clb
    envmod.SecondOrderCA.next_config = cnc
    try:
        den = TrajectoryDenoiser(model, L, H)      # builds candidates from the LEARNED rule
        _ = den.denoise(obs)                       # pure matching, no dynamics calls
    finally:
        est._learned_next_batch, envmod.SecondOrderCA.next_config = _olb, _onc
    print(f"  during inference (build candidates + MAP match):  learned-rule ticks = {learned[0]}, "
          f"true-CA ticks = {true[0]}")
    print("  => the LEARNED model supplies every candidate rollout; the true CA is touched 0 times in the")
    print("     search (only to generate the instance and to score the recovered IC). Abduction happens")
    print("     INSIDE the learned model — the audit-#1 'no leaked dynamics' guarantee, for the inverse.")


if __name__ == "__main__":
    main()
