"""CA substrate: a second-order *reversible* cellular automaton on a ring.

Rule 110 is computationally irreducible (class-4); Rule 90 is the reducible control. The
learned radius-1 local rule (``rule.py``) is trained on black-box transitions and never
reads the true rule; abduction is a MAP trajectory denoiser over candidate initial
conditions rolled forward in the learned rule (``estimate.py``). ``planner.py`` is the
light-cone forward simulator used by the denoiser; ``headtohead.py`` holds the LLM
baseline prompts and parsers.
"""
