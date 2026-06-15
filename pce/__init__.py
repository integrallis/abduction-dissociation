"""Learned-simulation reasoning.

A deliberately non-transformer reasoner that learns a world model from black-box
``(state, [action,] next-state)`` transitions and reasons by *simulating* that model --
forward execution, goal-directed control, and (the paper's focus) inverse inference /
abduction, recovering an unobserved initial condition from a noisy trajectory by MAP over
forward rollouts.

The package is intentionally flat; experiments import submodules directly, e.g.
``from pce.ca.env import RULE110``. See the paper in ``paper/`` and the README for the
claim-to-command map.
"""
