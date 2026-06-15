# Make the `pce` package importable when running the tests from the repo root,
# regardless of how pytest is invoked.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
