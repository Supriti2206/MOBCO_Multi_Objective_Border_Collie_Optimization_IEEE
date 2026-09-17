"""
dominance.py -- Direction-aware Pareto dominance utilities.

Every function here takes `objective_types` (a list like ['min', 'max', ...])
so that 'max' objectives are handled correctly without needing to sign-flip
the raw values beforehand (a design choice noted in the project history to
avoid sign-flipping tricks elsewhere in the codebase).

Pareto dominance definition used throughout: solution a dominates b iff a is
not worse than b in every objective AND strictly better in at least one.
"""

import numpy as np


def min_mask(objective_types):
    """Boolean array, True where the objective is to be minimised."""
    return np.asarray([t == 'min' for t in objective_types], dtype=bool)


def is_better_eq(a, b, mins):
    """Element-wise 'a is not worse than b', honouring each direction."""
    return np.where(mins, a <= b, a >= b)


def is_strictly_better(a, b, mins):
    """Element-wise 'a is strictly better than b', honouring each direction."""
    return np.where(mins, a < b, a > b)


def dominates(a, b, objective_types):
    """True if solution vector a dominates b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mins = min_mask(objective_types)
    return bool(np.all(is_better_eq(a, b, mins)) and
                np.any(is_strictly_better(a, b, mins)))


def dominance_matrix(F, objective_types):
    """
    D[i, j] == True  <=>  solution i dominates solution j.
    F : (N, M) array of raw objective values.
    """
    F = np.asarray(F, dtype=float)
    mins = min_mask(objective_types)
    # Broadcast every pair (i, j) at once: A varies over the first axis,
    # B over the second, so A <= B (etc.) evaluates all N x N comparisons
    # for all M objectives in one vectorized pass (no explicit double loop).
    A = F[:, None, :]                 # (N, 1, M)
    B = F[None, :, :]                 # (1, N, M)
    not_worse = np.where(mins, A <= B, A >= B)
    better = np.where(mins, A < B, A > B)
    D = not_worse.all(axis=2) & better.any(axis=2)  # dominance requires both conditions
    np.fill_diagonal(D, False)  # a solution never "dominates" itself
    return D


def domination_counts(F, objective_types):
    """
    Returns (strength, dominated_by) where
        strength[i]     = how many solutions i dominates
        dominated_by[i] = list of indices that dominate i
    """
    D = dominance_matrix(F, objective_types)
    strength = D.sum(axis=1)
    dominated_by = [np.flatnonzero(D[:, j]) for j in range(F.shape[0])]
    return strength, dominated_by


def nondominated_mask(F, objective_types):
    """Boolean mask of the non-dominated members of F."""
    D = dominance_matrix(F, objective_types)
    return ~D.any(axis=0)


def filter_nondominated(F, X=None, objective_types=None):
    """Return (F_nd, X_nd) keeping only non-dominated rows."""
    mask = nondominated_mask(F, objective_types)
    if X is None:
        return F[mask]
    return F[mask], X[mask]


def constraint_ok(x, bounds):
    """Box-feasibility check (all our problems are box constrained)."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    return bool(np.all(x >= lo - 1e-12) and np.all(x <= hi + 1e-12))
