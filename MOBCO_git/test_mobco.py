"""
test_mobco.py -- Minimal ad-hoc smoke test: run MOBCO once on Mixed_ZDT1
and print a few sanity-check numbers. Not a pytest/unittest suite (compare
to check.py, which is the project's actual structured self-check) -- just
a quick manual "does this even run" script, intended to be run directly
(`python test_mobco.py`) rather than collected by a test runner.

Note: cfg.n_runs=50 has no effect here -- this script calls MOBCO(...).run()
directly for a SINGLE run/seed, bypassing run_experiment.py's multi-run
loop entirely, so n_runs (and n_runs-dependent aggregation) is unused.
"""

import numpy as np
from dataset import get_problem
from optimizer import MOBCO
from config import MOBCOConfig

# Quick test
cfg = MOBCOConfig(n_population=100, n_dogs=3, archive_size=200,
                  max_iterations=200, n_runs=50)
problem = get_problem('Mixed_ZDT1')

mobco = MOBCO(problem, cfg, seed=42)
result = mobco.run()

print(f"Front size: {len(result['front'])}")
print(f"First 3 solutions:\n{result['front'][:3]}")
print(f"Function evaluations: {result['n_eval']}")
print(f"Runtime: {result['runtime']:.3f}s")