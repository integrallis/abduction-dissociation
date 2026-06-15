"""Sokoban substrate: a 2-D push world (irreversible dynamics, PSPACE-complete control).

The same learned-simulation reasoner as ``pce.ca`` over factored, multi-object state: one
weight-shared local push-rule learned from black-box transitions (``rule.py``), and
abduction by MAP over candidate initial boards rolled forward in the learned model
(``abduce.py``). ``headtohead.py`` holds the LLM baseline (ASCII rendering + parser).
"""
