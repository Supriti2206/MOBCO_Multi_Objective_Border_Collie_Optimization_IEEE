"""
nondominated_sort.py -- Fast non-dominated sorting, i.e.
partitioning a population into successive Pareto fronts:
front 0 = non-dominated solutions, front 1 = non-dominated once front 0 is
removed, and so on.
"""

import numpy as np

from dominance import dominance_matrix


def fast_nondominated_sort(F, objective_types):
    """
    Returns a list of fronts; fronts[0] is the non-dominated set.
    Each front is an array of indices into F.

    Implementation follows the classic NSGA-II approach: compute, for every
    solution, how many others dominate it (`dominated_count`); anything with
    a count of 0 forms the first front. Then simulate "removing" that front
    by decrementing the counts of everything it dominated, and repeat --
    whatever's count drops to 0 next becomes the following front.
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n == 0:
        return []

    D = dominance_matrix(F, objective_types)  # D[i, j] True <=> i dominates j
    dominated_count = D.sum(axis=0).astype(int)     # how many dominate i
    fronts = []
    current = np.flatnonzero(dominated_count == 0)  # front 0: nobody dominates these
    remaining = dominated_count.copy()
    assigned = np.zeros(n, dtype=bool)

    while current.size:
        fronts.append(current)
        assigned[current] = True
        remaining[current] = -1  # mark as already placed, so it's never re-selected
        # For each solution in the current front, count how many still-
        # unassigned solutions it dominates, and knock that many off their
        # remaining dominated_count.
        knocked = D[current].sum(axis=0)
        nxt = []
        for j in range(n):
            if assigned[j] or knocked[j] == 0:
                continue
            remaining[j] -= knocked[j]
            if remaining[j] <= 0:
                # All of j's dominators are now in earlier fronts -> j
                # belongs to the next front
                nxt.append(j)
        current = np.array(sorted(set(nxt)), dtype=int)
    return fronts


def rank_of_each(F, objective_types):
    """Front index (0-based) of every solution."""
    ranks = np.zeros(len(F), dtype=int)
    for r, front in enumerate(fast_nondominated_sort(F, objective_types)):
        ranks[front] = r
    return ranks


def pareto_front(F, X=None, objective_types=None):
    """Non-dominated subset (objectives and, optionally, decision vectors)."""
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return (F, X) if X is not None else F
    fronts = fast_nondominated_sort(F, objective_types)
    idx = fronts[0]
    if X is None:
        return F[idx]
    return F[idx], np.asarray(X)[idx]


def unique_rows(F, tol=1e-12):
    """Drop duplicate objective vectors (keeps the front tidy)."""
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return F, np.array([], dtype=int)
    order = np.lexsort(F.T[::-1])
    keep = [order[0]]
    for i in order[1:]:
        if np.any(np.abs(F[i] - F[keep[-1]]) > tol):
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return F[keep], keep
